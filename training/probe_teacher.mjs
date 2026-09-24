// How many streamed requests can the teacher server take at once before first bytes
// stop arriving within Cloudflare's ~100 s window?  node training/probe_teacher.mjs 1 4 8
const url = process.env.TEACHER_URL + "/v1/chat/completions"
const headers = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${process.env.TEACHER_KEY}`,
  "User-Agent": "jevforge-teacher/0.1",
}

async function one(i) {
  const t0 = Date.now()
  let first = null, chars = 0
  try {
    const res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: process.env.TEACHER_MODEL,
        messages: [{ role: "user", content: `Write about 300 words on why lighthouses were built where they were. (${i})` }],
        max_tokens: 600,
        stream: true,
        ...(process.env.TEACHER_NO_KWARGS ? {} : { chat_template_kwargs: { enable_thinking: true } }),
      }),
    })
    if (!res.ok) return { i, status: res.status, ms: Date.now() - t0 }
    for await (const chunk of res.body) {
      if (first === null) first = Date.now() - t0
      chars += chunk.length
    }
    return { i, status: 200, firstByteMs: first, totalMs: Date.now() - t0, bytes: chars }
  } catch (e) {
    return { i, error: String(e).slice(0, 80), ms: Date.now() - t0 }
  }
}

for (const n of process.argv.slice(2).map(Number)) {
  const results = await Promise.all(Array.from({ length: n }, (_, i) => one(i)))
  const ok = results.filter(r => r.status === 200)
  const fb = ok.map(r => r.firstByteMs).sort((a, b) => a - b)
  console.log(
    `${n} at once: ${ok.length}/${n} ok, first byte ${fb[0]}-${fb[fb.length - 1]} ms, ` +
      `total up to ${Math.max(...ok.map(r => r.totalMs), 0)} ms; failures: ` +
      JSON.stringify(results.filter(r => r.status !== 200).map(r => r.status || r.error))
  )
}
