// Live dashboard for an unattended run: teachers, the training round, GPU, and scores.
//   node dashboard/server.mjs [port]      -> http://localhost:8091
// Reads the files the run writes; changes nothing. Node built-ins only.
import { execFile } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const port = Number(process.argv[2] || 8091);
const at = (...p) => path.join(root, ...p);
const read = (p) => { try { return fs.readFileSync(p, "utf8"); } catch { return ""; } };
const run = (cmd, args) => new Promise((resolve) =>
  execFile(cmd, args, { timeout: 15000, windowsHide: true }, (_e, out, err) => resolve(`${out || ""}${err || ""}`)));

// ---- teacher files ---------------------------------------------------------------------------
const BASE_SOURCES = { "train.jsonl": "Remote 27B", "train-local.jsonl": "Local 27B", "smoke.jsonl": "Smoke test" };
// Rented pods write train-pod.jsonl, train-pod2.jsonl ... and show up here as soon as a file exists.
function sources() {
  const out = { ...BASE_SOURCES };
  let files = [];
  try { files = fs.readdirSync(at("data", "teacher")); } catch {}
  // Pods named in h100-env.sh get a card before their first document (only the POD*_ID lines are read).
  for (const m of read(at("h100-env.sh")).matchAll(/^export POD(\d*)_ID=/gm)) files.push(`train-pod${m[1]}.jsonl`);
  files = [...new Set(files)];
  for (const f of files.sort((a, b) => (+a.match(/\d+/)?.[0] || 1) - (+b.match(/\d+/)?.[0] || 1))) {
    const m = f.match(/^train-pod(\d*)\.jsonl$/);
    if (m) out[f] = m[1] ? `Runpod 27B #${m[1]}` : "Runpod 27B";
  }
  return out;
}
const cache = new Map();
function teacherFile(name) {
  const p = at("data", "teacher", name);
  let st; try { st = fs.statSync(p); } catch { return null; }
  const hit = cache.get(name);
  if (hit && hit.mtime === st.mtimeMs) return hit.data;
  const data = { docs: new Set(), questions: 0, kept: 0, errors: {}, families: {}, types: {}, recent: [], tokens: 0 };
  for (const line of read(p).split("\n")) {
    if (!line.trim()) continue;
    let r; try { r = JSON.parse(line); } catch { continue; }
    data.docs.add(r.doc);
    if (r.error) {
      const k = r.error.replace(/HTTPError: /, "").slice(0, 44);
      data.errors[k] = (data.errors[k] || 0) + 1;
      continue;
    }
    data.questions++;
    data.tokens += (r.author_tokens || 0) + (r.solve_tokens || 0);
    if (!r.agree) continue;
    data.kept++;
    const fam = r.spec?.family || "other";
    data.families[fam] = (data.families[fam] || 0) + 1;
    const t = r.question?.type || "?";
    data.types[t] = (data.types[t] || 0) + 1;
    data.recent.push({
      family: fam, domain: r.spec?.domain || "", type: t,
      q: r.question?.instructions || "", answer: r.expected,
      options: Object.keys(r.question?.criteria || {}).length, source: sources()[name],
    });
  }
  data.recent = data.recent.slice(-8);
  data.docs = data.docs.size;
  cache.set(name, { mtime: st.mtimeMs, data });
  return data;
}

// ---- kept-over-time history, sampled here since records carry no timestamps -------------------
const histPath = at("results", "dashboard-history.json");
let history = [];
try { history = JSON.parse(read(histPath)) || []; } catch { history = []; }
function sample() {
  const row = { t: Date.now() };
  for (const f of Object.keys(sources())) row[f] = teacherFile(f)?.kept || 0;
  const last = history[history.length - 1];
  if (!last || row.t - last.t >= 55000) {
    history.push(row);
    history = history.slice(-2000);
    try { fs.writeFileSync(histPath, JSON.stringify(history)); } catch {}
  }
}
sample();
setInterval(sample, 60000);

// ---- slow probes, cached ---------------------------------------------------------------------
let probes = { gpu: null, slots: [], processes: {}, at: 0 };
async function probe() {
  const [gpu, logs, running, procs] = await Promise.all([
    run("nvidia-smi", ["--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
      "--format=csv,noheader,nounits"]),
    run("docker", ["logs", "--tail", "30", "jevy-teacher"]),
    run("docker", ["ps", "--filter", "name=jevy-teacher", "--format", "{{.Names}}"]),
    run("powershell", ["-NoProfile", "-Command",
      "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -and $_.CommandLine -match 'teacher\\.py|lora\\.py|jevbench|fit_temperature') -or ($_.Name -eq 'bash.exe' -and $_.CommandLine -match 'supervise\\.sh|experiments\\.sh') -or ($_.Name -eq 'docker.exe' -and $_.CommandLine -match 'jevy:latest') } | ForEach-Object { $_.CreationDate.ToString('o') + '|' + $_.CommandLine }"]),
  ]);
  const g = gpu.trim().split(",").map((s) => s.trim());
  const slots = {};
  for (const m of logs.matchAll(/id\s+(\d+) \| task \d+ \| n_gen =\s*(\d+), tg =\s*([\d.]+) t\/s/g)) slots[m[1]] = { gen: +m[2], tps: +m[3] };
  probes = {
    gpu: g.length >= 6 && !isNaN(+g[1]) ? { name: g[0], util: +g[1], used: +g[2], total: +g[3], temp: +g[4], watts: +g[5] } : null,
    slots: Object.entries(slots).map(([id, s]) => ({ id: +id, ...s })),
    server: /jevy-teacher/.test(running),
    processes: {
      supervisor: /supervise\.sh|experiments\.sh/.test(procs),
      remote: /teacher\.py.*--out data\/teacher\/train\.jsonl/.test(procs),
      local: /teacher\.py.*--out data\/teacher\/train-local\.jsonl/.test(procs),
      pod: /teacher\.py.*--out data\/teacher\/train-pod\.jsonl/.test(procs),
      pod2: /teacher\.py.*--out data\/teacher\/train-pod2\.jsonl/.test(procs),
      pod3: /teacher\.py.*--out data\/teacher\/train-pod3\.jsonl/.test(procs),
      container: /jevy:latest/.test(procs),
      byFile: Object.fromEntries(Object.keys(sources()).map((f) => [f, procs.includes(`--out data/teacher/${f}`)])),
    },
    at: Date.now(),
  };
  // When each teacher process started, and when each source's document count last went up.
  for (const line of procs.split(/\r?\n/)) {
    const m = line.match(/^([^|]+)\|.*teacher\.py.*--out data\/teacher\/(\S+\.jsonl)/i);
    if (m) startedAt[m[2]] = Math.max(startedAt[m[2]] || 0, new Date(m[1]).getTime());
  }
  for (const f of Object.keys(sources())) {
    const docs = teacherFile(f)?.docs || 0;
    if (f in lastDocs && docs > lastDocs[f]) {
      (landed[f] ??= []).push({ t: Date.now(), n: docs - lastDocs[f] });
      landed[f] = landed[f].slice(-40);
    }
    lastDocs[f] = docs;
  }
}
// A restarted teacher's first documents took 12-14 minutes today (every stream starts together);
// after that, documents land continuously, so the recent average gap predicts the next one.
const FIRST_WAVE = 14 * 60e3;
const startedAt = {}, landed = {}, lastDocs = {};
function nextDocument(f) {
  const start = startedAt[f];
  if (!start) return null;
  const since = (landed[f] || []).filter((e) => e.t >= start);
  let written = 0;
  try { written = fs.statSync(at("data", "teacher", f)).mtimeMs; } catch {}
  if (!since.length && written <= start) return { at: start + FIRST_WAVE, basis: "first documents after a restart" };
  if (since.length < 3) return { at: Math.max(written, since.at(-1)?.t || 0) + 90e3, basis: "measuring the pace" };
  const recent = since.slice(-12), last = recent[recent.length - 1];
  const docs = recent.slice(1).reduce((n, e) => n + e.n, 0);
  const gap = (last.t - recent[0].t) / Math.max(1, docs);
  return { at: last.t + gap, gap, basis: `one every ${Math.round(gap / 1000)} s lately` };
}
probe();
setInterval(probe, 10000);

// ---- supervisor, round and scores ------------------------------------------------------------
function supervisor() {
  const log = read(at("results", "supervise.log")).split(/\r?\n/).filter(Boolean);
  const startLine = [...log].reverse().find((l) => / start: ([\d.]+h|until \d)/.test(l));
  let start = null, deadline = null, target = 400;
  if (startLine) {
    start = new Date(startLine.slice(0, 19).replace(" ", "T")).getTime();
    const hours = startLine.match(/start: ([\d.]+)h/), until = startLine.match(/until (\d+):(\d+)/);
    if (hours) deadline = start + parseFloat(hours[1]) * 3600e3;
    else { const d = new Date(start); d.setHours(+until[1], +until[2], 0, 0); deadline = d.getTime() < start ? d.getTime() + 864e5 : d.getTime(); }
  }
  const sinceStart = startLine ? log.slice(log.lastIndexOf(startLine)) : log;
  for (const l of sinceStart) { const m = l.match(/round at (\d+)/); if (m) target = +m[1]; }
  const every = +(startLine?.match(/then every (\d+) more/)?.[1] || 300);
  let round = null;
  for (const l of sinceStart) {
    const m = l.match(/round (v\w+): training on (\d+)/);
    if (m) round = { name: m[1], kept: +m[2], step: "starting", done: false, failed: false };
    if (round && l.includes("== ")) round.step = l.replace(/^.*== /, "");
    if (round && /pushed:|done v\d/.test(l)) { round.done = true; round.step = "pushed"; target = round.kept + every; }
    if (round && /failed; see/.test(l)) { round.failed = true; round.done = true; round.step = "failed"; target = round.kept + every; }
  }
  if (round) {
    const tl = read(at("results", `${round.name}-train.log`)).replace(/\r/g, "\n");
    const steps = [...tl.matchAll(/step (\d+)\/(\d+) loss ([\d.]+) ~(\d+) min left/g)];
    if (steps.length) {
      const s = steps[steps.length - 1];
      round.train = { step: +s[1], of: +s[2], loss: +s[3], minLeft: +s[4], losses: steps.map((x) => +x[3]) };
    }
    const b = tl.match(/before (\{.*\})/), a = tl.match(/after (\{.*\})/);
    if (b) try { round.before = JSON.parse(b[1]); } catch {}
    if (a) try { round.after = JSON.parse(a[1]); } catch {}
    const bl = read(at("results", `jevy-${round.name}.log`));
    const splits = (bl.match(/\[jevbench\] done/g) || []).length;
    const done = [...bl.matchAll(/(\d+)\/(\d+) completed/g)].pop();
    if (bl) round.bench = { splitsDone: splits, current: done ? `${done[1]}/${done[2]}` : null };
    const temp = sinceStart.map((l) => l.match(/^temperature ([\d.]+)/)).filter(Boolean).pop();
    if (temp) round.temperature = +temp[1];
    round.passes = [...tl.matchAll(/^pass (\d+) (\{.*\})/gm)].map((m) => ({ pass: +m[1], ...JSON.parse(m[2]) }));
    const kept = tl.match(/^keeping pass (\d+)/m);
    if (kept) round.kept_pass = +kept[1];
  }
  const decided = log.some((l) => /deadline reached/.test(l) && startLine && log.indexOf(l) > log.lastIndexOf(startLine));
  let n = 2;
  while (fs.existsSync(at("results", `jevy-v${n}`))) n++;
  // every round started since the last supervisor/experiments start, with its held-out and public results
  const rounds = [];
  for (const l of sinceStart) {
    const m = l.match(/round (v\w+): training on (\d+) kept teacher questions(?: \((.*)\))?/);
    if (!m) continue;
    const name = m[1], tl = read(at("results", `${name}-train.log`)).replace(/\r/g, "\n");
    const after = tl.match(/^after (\{.*\})/m), kept = tl.match(/^keeping pass (\d+)/m);
    const sc = read(at("results", `jevy-${name}`, "SCORES.txt")).split(/\r?\n/)
      .find((x) => new RegExp(`^jevy-${name}\\s+hard\\s`).test(x));
    const hard = sc ? sc.trim().split(/\s+/) : null;
    rounds.push({ name, what: m[3] || "", after: after ? JSON.parse(after[1]).acc : null,
      kept: kept ? +kept[1] : null, hard: hard ? +hard[3] : null, hardEce: hard ? +hard[4] : null,
      failed: sinceStart.some((x) => x.includes(`round ${name} failed`)) });
  }
  return { start, deadline, target, round, rounds, nextRound: `v${n}`, finished: decided, tail: log.slice(-14) };
}

function scores() {
  const runs = {};
  let dirs = [];
  try { dirs = fs.readdirSync(at("results"), { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name); } catch {}
  for (const d of dirs) {
    for (const line of read(at("results", d, "SCORES.txt")).split(/\r?\n/).slice(1)) {
      const p = line.trim().split(/\s+/);
      if (p.length < 7) continue;
      runs[p[0]] ??= {};
      runs[p[0]][p[1]] = { n: +p[2], acc: +p[3], ece: +p[4], p50: +p[5] };
    }
  }
  return runs;
}

function status() {
  const teachers = {};
  for (const f of Object.keys(sources())) { const t = teacherFile(f); if (t) teachers[sources()[f]] = t; }
  const total = Object.values(teachers).reduce((s, t) => s + t.kept, 0);
  const hourAgo = [...history].reverse().find((h) => Date.now() - h.t >= 3600e3) || history[0];
  const sum = (h) => Object.keys(sources()).reduce((s, f) => s + (h?.[f] || 0), 0);
  const spanH = hourAgo ? (Date.now() - hourAgo.t) / 3600e3 : 0;
  const perSource = {};
  for (const [f, label] of Object.entries(sources())) {
    perSource[label] = spanH > 0.05 ? ((teachers[label]?.kept || 0) - (hourAgo?.[f] || 0)) / spanH : null;
  }
  return {
    now: Date.now(), total, teachers,
    rate: spanH > 0.05 ? (total - sum(hourAgo)) / spanH : null, perSource, rateSpanMin: Math.round(spanH * 60),
    history: history.map((h) => [h.t, sum(h)]),
    teacherList: Object.entries(sources()).map(([file, label]) => ({ file, label })),
    etas: Object.fromEntries(Object.entries(sources()).map(([f, label]) => [label, nextDocument(f)])),
    probes, supervisor: supervisor(), scores: scores(),
  };
}

http.createServer((req, res) => {
  if (req.url.startsWith("/api/status")) {
    res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
    res.end(JSON.stringify(status()));
    return;
  }
  const doc = req.url.match(/^\/docs\/([\w.-]+\.(svg|html))$/);
  if (doc) {
    const body = read(at("docs", doc[1]));
    res.writeHead(body ? 200 : 404, { "content-type": doc[2] === "svg" ? "image/svg+xml" : "text/html; charset=utf-8" });
    res.end(body);
    return;
  }
  res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
  res.end(read(at("dashboard", "index.html")));
}).listen(port, "127.0.0.1", () => console.log(`jevy dashboard on http://localhost:${port}`));
