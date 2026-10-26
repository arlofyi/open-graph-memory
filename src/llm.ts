// Model access. Chat goes through any OpenAI-compatible chat-completions
// endpoint, set with OGM_CHAT_BASE (and OGM_CHAT_KEY if it needs a key).
// Embeddings come from a local Ollama bge-m3, cached on disk by content hash
// so reruns are free.

import { createHash } from "node:crypto"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"

export const CHAT_BASE = process.env.OGM_CHAT_BASE
const CHAT_KEY = process.env.OGM_CHAT_KEY ?? "local"
export const EMBED_BASE = process.env.OGM_EMBED_BASE ?? "http://127.0.0.1:11434"
export const EMBED_MODEL = process.env.OGM_EMBED_MODEL ?? "bge-m3"
export const CACHE = process.env.OGM_CACHE ?? join(import.meta.dirname, "..", ".cache")

export const sha1 = (s: string): string => createHash("sha1").update(s).digest("hex")

export interface ChatOptions { json?: boolean; maxTokens?: number }

export async function chat(model: string, messages: Array<{ role: string; content: string }>, opts: ChatOptions = {}): Promise<string> {
  if (!CHAT_BASE) throw new Error("set OGM_CHAT_BASE to an OpenAI-compatible endpoint, e.g. http://127.0.0.1:4000/v1")
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(`${CHAT_BASE}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${CHAT_KEY}` },
      body: JSON.stringify({
        model,
        messages,
        max_tokens: opts.maxTokens ?? 4000,
        ...(opts.json ? { response_format: { type: "json_object" } } : {}),
      }),
    })
    const body = await res.text()
    if (res.ok) {
      const j = JSON.parse(body) as { choices: Array<{ message: { content: string } }> }
      return j.choices[0]?.message.content ?? ""
    }
    // A metering proxy's spend-limit refusal is final; transient upstream
    // errors get three retries.
    if (body.includes("budget_exceeded") || attempt >= 3 || res.status < 500) throw new Error(`chat ${res.status}: ${body.slice(0, 300)}`)
    await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)))
  }
}

export function parseJson<T>(raw: string): T {
  const start = raw.indexOf("{")
  const end = raw.lastIndexOf("}")
  return JSON.parse(start >= 0 && end > start ? raw.slice(start, end + 1) : raw) as T
}

const EMB_DIR = join(CACHE, "emb", EMBED_MODEL)

function embPath(text: string): string {
  const h = sha1(text)
  return join(EMB_DIR, h.slice(0, 2), `${h}.f32`)
}

/** Read a cached vector. readFileSync may return a view into a pooled 64 KB
 *  ArrayBuffer, so copy exactly [byteOffset, byteOffset + length). */
function readVec(p: string): Float32Array {
  const b = readFileSync(p)
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length))
}

function normalise(v: Float32Array): Float32Array {
  let n = 0
  for (let i = 0; i < v.length; i++) n += v[i]! * v[i]!
  n = Math.sqrt(n) || 1
  for (let i = 0; i < v.length; i++) v[i]! /= n
  return v
}

/** Unit vectors for `texts`, batched (32) and cached on disk. */
export async function embedMany(texts: string[], log?: (m: string) => void): Promise<Float32Array[]> {
  const out: Array<Float32Array | undefined> = texts.map((t) => {
    const p = embPath(t)
    return existsSync(p) ? readVec(p) : undefined
  })
  const missing = [...new Set(texts.filter((_, i) => !out[i]))]
  for (let i = 0; i < missing.length; i += 32) {
    const batch = missing.slice(i, i + 32)
    const res = await fetch(`${EMBED_BASE}/api/embed`, { method: "POST", body: JSON.stringify({ model: EMBED_MODEL, input: batch }) })
    if (!res.ok) throw new Error(`embed ${res.status}: ${(await res.text()).slice(0, 200)}`)
    const j = (await res.json()) as { embeddings: number[][] }
    batch.forEach((t, bi) => {
      const v = normalise(Float32Array.from(j.embeddings[bi]!))
      const p = embPath(t)
      mkdirSync(join(p, ".."), { recursive: true })
      writeFileSync(p, Buffer.from(v.buffer))
    })
    if (log && (i / 32) % 20 === 0) log(`  embedded ${Math.min(i + 32, missing.length)}/${missing.length}`)
  }
  return texts.map((t, i) => out[i] ?? readVec(embPath(t)))
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!
  return s
}

/** Run `fn` over `items` with at most `n` in flight, preserving order. */
export async function mapLimit<T, R>(items: T[], n: number, fn: (x: T, i: number) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length)
  let next = 0
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => {
    while (next < items.length) {
      const i = next++
      out[i] = await fn(items[i]!, i)
    }
  }))
  return out
}
