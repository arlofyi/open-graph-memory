// LoCoMo loader (snap-research/locomo, CC BY-NC 4.0). The dataset is never
// committed: it is read from .cache/locomo/locomo10.json (gitignored), or from
// $LOCOMO_PATH. Turn rendering, session order, splits and scored categories
// follow arlo-memory's benchmark harness, so numbers are comparable with the
// arlo-memory baselines quoted in docs/RESULTS.md.

import { readFileSync } from "node:fs"
import { join } from "node:path"

export const LOCOMO_PATH = process.env.LOCOMO_PATH ?? join(import.meta.dirname, "..", ".cache", "locomo", "locomo10.json")
export const DEV = [0, 1]
export const TEST = [2, 3, 4, 5, 6, 7, 8, 9]
export const CATEGORY: Record<number, string> = { 1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop" }

export interface Turn {
  id: string // canonical dia id, "D3:5"
  session: string // "session_3"
  sessionIndex: number // chronological position
  date: Date
  speaker: string
  text: string // "Speaker: text [Sharing image …]"
  prev?: string // previous turn in the same session, rendered
}

export interface Question {
  qid: string
  conv: number
  category: number
  question: string
  answer: string
  evidence: string[] // canonical dia ids that exist as turns
}

export interface Conversation {
  index: number
  speakers: string[]
  sessions: Array<{ key: string; date: Date; rawDate: string; turns: Turn[] }>
  turns: Turn[]
  questions: Question[]
}

const MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

export function parseDate(raw: string): Date {
  const m = /^\s*(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\s*$/i.exec(raw)
  if (!m) throw new Error(`unparseable LoCoMo date "${raw}"`)
  let hour = Number(m[1]) % 12
  if (m[3]!.toLowerCase() === "pm") hour += 12
  const mn = m[5]!.toLowerCase()
  const month = MONTHS.findIndex((n) => n === mn || n.slice(0, 3) === mn.slice(0, 3))
  return new Date(Date.UTC(Number(m[6]), month, Number(m[4]), hour, Number(m[2])))
}

export function canonicalDiaId(raw: string): string | null {
  const m = /D\s*:?\s*(\d+)\s*:\s*(\d+)/i.exec(raw)
  return m ? `D${Number(m[1])}:${Number(m[2])}` : null
}

interface RawTurn { speaker: string; dia_id?: string; text?: string; blip_caption?: string; query?: string }
interface RawConv { qa: Array<{ question: string; answer?: string | number; evidence?: string[]; category: number }>; conversation: Record<string, unknown> }

function turnText(t: RawTurn): string {
  const text = (t.text ?? "").trim()
  const blip = (t.blip_caption ?? "").trim()
  const query = (t.query ?? "").trim()
  let tag = ""
  if (query && blip) tag = `[Sharing image - query: ${query}. The image shows: ${blip}]`
  else if (query) tag = `[Sharing image - query for: ${query}]`
  else if (blip) tag = `[Sharing image that shows: ${blip}]`
  return tag ? (text ? `${text} ${tag}` : tag) : text
}

export function loadLocomo(path = LOCOMO_PATH): Conversation[] {
  const data = JSON.parse(readFileSync(path, "utf8")) as RawConv[]
  if (!Array.isArray(data) || data.length !== 10) throw new Error(`expected 10 conversations in ${path}`)
  return data.map((raw, index) => {
    const c = raw.conversation
    const sessions = Object.keys(c)
      .map((key) => /^session_(\d+)$/.exec(key))
      .filter((m): m is RegExpExecArray => !!m && Array.isArray(c[m[0]]))
      .map((m) => ({ key: m[0], number: Number(m[1]), rawDate: String(c[`${m[0]}_date_time`] ?? "") }))
      .map((s) => ({ ...s, date: parseDate(s.rawDate) }))
      .sort((a, b) => a.date.getTime() - b.date.getTime() || a.number - b.number)
    const turns: Turn[] = []
    const out = sessions.map((s, sessionIndex) => {
      const st: Turn[] = []
      let prev: string | undefined
      for (const t of c[s.key] as RawTurn[]) {
        const text = turnText(t)
        const id = t.dia_id ? canonicalDiaId(t.dia_id) : null
        if (!text || !id) continue
        const rendered = `${t.speaker}: ${text}`
        st.push({ id, session: s.key, sessionIndex, date: s.date, speaker: t.speaker, text: rendered, ...(prev ? { prev } : {}) })
        prev = rendered
      }
      turns.push(...st)
      return { key: s.key, date: s.date, rawDate: s.rawDate, turns: st }
    })
    const ids = new Set(turns.map((t) => t.id))
    const questions: Question[] = []
    raw.qa.forEach((qa, qi) => {
      if (!(qa.category in CATEGORY)) return
      const evidence = [...new Set((qa.evidence ?? []).map(canonicalDiaId).filter((x): x is string => !!x && ids.has(x)))]
      let answer = String(qa.answer ?? "")
      if (qa.category === 3) answer = answer.split(";")[0]!.trim()
      questions.push({ qid: `conv${index}_q${qi}`, conv: index, category: qa.category, question: qa.question, answer, evidence })
    })
    const speakers = [String(c.speaker_a ?? ""), String(c.speaker_b ?? "")].filter(Boolean)
    return { index, speakers, sessions: out, turns, questions }
  })
}
