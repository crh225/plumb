"""Plumb-4B server: typed decisions over TypeSafe's /v1/systemone wire format, one forward pass per decision.

    python plumb_server.py --model ./plumb-4b --port 8090 --one-read --temperature 2.07 \
        --noul-commit --score-temperature 1.2          # Plumb-4B v5.2, as submitted to JevBench v1.5

v5.2 uses the v5 weights unchanged and reads each question once, with two settings for JevBench v1.5's scoring rules:
--noul-commit reports yes/no answers whose P(yes) falls inside the 0.2-0.8 non-credit band at the nearest edge
(0.801 / 0.199), keeping the model's own decision; answers already outside the band are unchanged. Score questions
are read at --score-temperature (1.2) instead of the choice temperature (2.07). Both settings were chosen on Plumb's
own development data, never on JevBench items. usage.input_tokens counts every token processed.

Without --one-read, the server runs the experimental v5.1 readout: when the top probability is below TAU it also
reads three other option orders and averages them (the shared evidence prefix is read once).

The prompt and letter readout follow jevk5 v0.2.0 (Apache-2.0, github.com/allebee/jevk5), which follows
SemIf (MIT): a fixed system instruction, the decision as JSON, the chat template with thinking off, and a
softmax over the answer letters' next-token logits.
"""

from __future__ import annotations

import argparse
import copy
import http.server
import json
import threading

import numpy as np
import torch

T1, T4, TAU = 2.28, 1.88, 0.5
COMMIT_EDGE = 0.801  # just past JevBench v1.5's 0.8 yes/no boundary
LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def messages(state, criterion: str, options: list[str]) -> list[dict]:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)],
    }
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def decision_options(question: dict) -> list[tuple[str, str]]:
    crit = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif question["type"] == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in pairs]


def orders(n: int) -> list[list[int]]:
    base = list(range(n))
    half = base[(n + 1) // 2 :] + base[: (n + 1) // 2]
    return [base, base[::-1], half, half[::-1]]


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class Plumb51:
    def __init__(self, source: str, device: str = "cuda", t1: float = T1, t4: float = T4, tau: float = TAU) -> None:
        import transformers

        config = transformers.AutoConfig.from_pretrained(source)
        self.tok = transformers.AutoTokenizer.from_pretrained(source)
        cls = transformers.AutoModelForCausalLM
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            cls, config = transformers.Qwen3_5ForCausalLM, config.get_text_config()
        self.model = cls.from_pretrained(source, config=config, dtype=torch.bfloat16, device_map={"": device}).eval()
        self.device, self.t1, self.t4, self.tau = device, t1, t4, tau
        self.noul_mode, self.noul_bias = "true_first", 0.0  # yes/no readout: true_first | false_first | average
        self.score_mode = "as_given"  # score readout: as_given | reversed | average (of as given and reversed)
        self.noul_commit, self.score_t = False, None  # v1.5 readout: commit in-band yes/no answers; score temperature
        slots = [self.tok.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(ids) != 1 for ids in slots):
            raise ValueError("Every answer letter must be one token")
        self.slot_weight = self.model.lm_head.weight[[ids[0] for ids in slots]].detach().contiguous()

    def encode(self, state, criterion: str, options: list[str]) -> list[int]:
        prompt = self.tok.apply_chat_template(messages(state, criterion, options), tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        return self.tok.encode(prompt, add_special_tokens=False)

    @torch.inference_mode()
    def _run(self, ids: list[int], past=None):
        out = self.model.model(input_ids=torch.tensor([ids], device=self.device), past_key_values=past, use_cache=past is not None)
        return (out.last_hidden_state[0, -1] @ self.slot_weight.T).float().cpu().numpy()

    @torch.inference_mode()
    def _prefix(self, ids: list[int]):
        return self.model.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True).past_key_values

    def read(self, state, question: dict, force_all: bool = False) -> dict:
        """Probabilities per option id, tokens processed, whether it looked twice, and the raw logits."""
        options = decision_options(question)
        yn = question["type"] == "noul"
        sc = question["type"] == "score"
        if (yn and self.noul_mode in ("false_first", "average")) or (sc and self.score_mode == "reversed"):
            options = options[::-1]  # yes/no: "false" first, as the benchmark lists "no" before "yes"; score: highest first
        n = len(options)
        perms = orders(n)
        seqs = [self.encode(state, question["instructions"], [options[j][1] for j in p]) for p in perms]
        first = self._run(seqs[0])[:n]
        tokens = len(seqs[0])
        if (yn and self.noul_mode == "average") or (sc and self.score_mode == "average"):
            second = self._run(seqs[1])[:n][::-1]  # the reversed order, mapped back
            first = (first + second) / 2
            tokens += len(seqs[1])
        if yn and self.noul_bias:
            first = first.copy()
            first[[k for k, _ in options].index("true")] -= self.noul_bias
        p1 = softmax(first / (self.score_t if sc and self.score_t else self.t1))
        if yn and self.noul_commit:
            # JevBench v1.5 counts 0.2 < P(yes) < 0.8 as an abstention; commit such answers to the nearest edge,
            # keeping the argmax decision. Answers already outside the band are unchanged.
            ti = [k for k, _ in options].index("true")
            pt = float(p1[ti])
            if 1 - COMMIT_EDGE < pt < COMMIT_EDGE:
                pt = COMMIT_EDGE if pt >= 0.5 else 1 - COMMIT_EDGE
                p1 = np.array([pt if i == ti else 1 - pt for i in range(n)])
        if (yn and (self.noul_mode != "true_first" or self.noul_bias or self.noul_commit)) or \
                (sc and (self.score_mode != "as_given" or self.score_t)):
            return {"probs": dict(zip([k for k, _ in options], p1.tolist())), "tokens": tokens, "second": False,
                    "logits": [first.tolist()]}
        if p1.max() >= self.tau and not force_all:
            return {"probs": dict(zip([k for k, _ in options], p1.tolist())), "tokens": len(seqs[0]), "second": False,
                    "logits": [first.tolist()]}
        k = 0
        while all(len(s) > k + 1 for s in seqs) and len({s[k] for s in seqs}) == 1:
            k += 1
        past = self._prefix(seqs[0][:k])
        logits, tokens = [first], len(seqs[0]) + k
        for p, s in zip(perms[1:], seqs[1:]):
            z = self._run(s[k:], copy.deepcopy(past))[:n]
            back = np.zeros(n)
            back[p] = z
            logits.append(back)
            tokens += len(s) - k
        p4 = softmax(np.mean(logits, 0) / self.t4)
        return {"probs": dict(zip([k_ for k_, _ in options], p4.tolist())), "tokens": tokens, "second": True,
                "logits": [z.tolist() for z in logits], "prefix": k}

    def decide(self, state, question: dict) -> dict:
        r = self.read(state, question)
        probs, kind = r["probs"], question["type"]
        answer = {"type": kind, "confidence": max(probs.values()), "input_tokens": r["tokens"]}
        if kind == "noul":
            answer["noul"] = probs["true"]
        elif kind == "choice":
            answer.update(choice=max(probs, key=probs.get), probabilities=probs)
        else:
            answer.update(score=sum(int(k) * v for k, v in probs.items()), probabilities=probs)
        return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", default="plumb-4b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--one-read", action="store_true", help="never look twice (v5 as submitted, T 2.07)")
    parser.add_argument("--temperature", type=float, default=2.07, help="one-read temperature")
    parser.add_argument("--noul-mode", choices=["true_first", "false_first", "average"], default="true_first",
                        help="yes/no readout: option order, or the mean of both orders")
    parser.add_argument("--noul-bias", type=float, default=0.0, help="subtracted from the 'true' logit on yes/no questions")
    parser.add_argument("--score-mode", choices=["as_given", "reversed", "average"], default="as_given",
                        help="score readout: level order, or the mean of both orders")
    parser.add_argument("--noul-commit", action="store_true",
                        help="yes/no answers with 0.2 < P(yes) < 0.8 are reported at 0.801 / 0.199 (JevBench v1.5 abstention band)")
    parser.add_argument("--score-temperature", type=float, default=None, help="temperature for score questions only")
    args = parser.parse_args()
    model = Plumb51(args.model, t1=args.temperature, tau=-1.0) if args.one_read else Plumb51(args.model)
    model.noul_mode, model.noul_bias, model.score_mode = args.noul_mode, args.noul_bias, args.score_mode
    model.noul_commit, model.score_t = args.noul_commit, args.score_temperature
    gpu = threading.Lock()
    model.decide("warm-up", {"type": "noul", "instructions": "Is this a warm-up?"})
    print(f"ready: {args.name} on {args.host}:{args.port}, T1 {model.t1} T4 {model.t4} tau {model.tau}", flush=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self.reply(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                state, questions = req["state"], req["questions"]
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("questions must be a non-empty object")
            except (ValueError, KeyError, TypeError) as e:
                return self.reply(400, {"error": f"bad request: {e}"})
            answers, tokens = {}, 0
            with gpu:
                for key, q in questions.items():
                    try:
                        q = {**q, "instructions": q.get("instructions", "")}
                        opts = decision_options(q)
                        if len(opts) > 16:
                            return self.reply(400, {"error": f"question {key}: more than 16 options"})
                        n = len(model.encode(state, q["instructions"], [t for _, t in opts]))
                        if n > args.max_tokens:
                            return self.reply(413, {"error": f"question {key}: {n} input tokens > {args.max_tokens}"})
                        answers[key] = model.decide(state, q)
                        tokens += answers[key]["input_tokens"]
                    except (KeyError, TypeError, ValueError) as e:
                        return self.reply(400, {"error": f"question {key}: {e}"})
            self.reply(200, {"model": args.name, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}})

    http.server.ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
