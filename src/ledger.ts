// The claim ledger: the only source of truth. Append-only; a correction is a
// new claim (kind "retract" / "same_as" / "not_same_as"), never an edit.
// The PoC keeps it as JSONL. Postgres (tstzrange + GiST EXCLUDE) is the target.

import { appendFileSync, existsSync, readFileSync } from "node:fs"

export type ClaimKind = "assert" | "retract" | "same_as" | "not_same_as"
export type ActorType = "user" | "machine" | "ai"

export interface Claim {
  id: string
  kind: ClaimKind
  subject: string // surface mention, as written by the extractor
  predicate: string
  object: string | null // mention or literal
  qualifiers: Record<string, string>
  text: string // standalone sentence
  /** When it was true in the world: ISO date / "YYYY-MM" / "YYYY", or null when unknown. */
  validFrom: string | null
  validTo: string | null
  /** When we learned it (the source turn's session time). */
  recordedAt: string
  sources: string[] // source ids (LoCoMo turn ids)
  confidence: number
  actor: ActorType
  extractor: string
  retracts?: string
}

export class Ledger {
  readonly claims: Claim[] = []
  readonly path: string | undefined
  constructor(path?: string) {
    this.path = path
    if (path && existsSync(path)) {
      for (const line of readFileSync(path, "utf8").split("\n")) if (line.trim()) this.claims.push(JSON.parse(line) as Claim)
    }
  }
  append(c: Claim): void {
    if (!c.sources.length) throw new Error(`claim ${c.id} has no source`) // no source, no claim
    this.claims.push(c)
    if (this.path) appendFileSync(this.path, `${JSON.stringify(c)}\n`)
  }
  /** Asserted claims not retracted as of recorded time `asOf` (ISO), or now. */
  live(asOf?: string): Claim[] {
    const t = asOf ?? "9999"
    const retracted = new Set(this.claims.filter((c) => c.kind === "retract" && c.recordedAt <= t).map((c) => c.retracts))
    return this.claims.filter((c) => c.kind === "assert" && c.recordedAt <= t && !retracted.has(c.id))
  }
}
