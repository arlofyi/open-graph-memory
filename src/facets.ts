// Facets: a coarse, typed view over claims, so a question can be answered by
// TRAVERSAL ("every activity claim about Melanie") instead of similarity alone.
// Tags are a projection like everything else: re-derivable, cached, versioned.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { CACHE, chat, mapLimit, parseJson, sha1 } from "./llm.ts"
import type { Claim } from "./ledger.ts"

export const FACETS = [
  "activity", // hobbies, sports, things done for fun or routine
  "event", // things attended or that happened: parades, trips, talks, accidents
  "travel_place", // places lived, visited, moved from
  "family", // relatives, kids, partners, family events
  "relationship", // friends, dating, breakups, relationship status
  "pet",
  "work_career", // jobs, career plans, business
  "education", // school, courses, classes, research
  "health", // physical or mental health, injuries
  "possession", // things owned, bought, received as gifts
  "preference", // likes, dislikes, favourites
  "belief_value", // opinions, values, identity-related beliefs
  "identity", // who someone is: gender, nationality, religion, background
  "plan_goal", // intentions, dreams, future plans
  "creative_work", // art, books, music, writing, made or consumed
  "feeling", // emotions about something
] as const
export type Facet = (typeof FACETS)[number]
export const FACET_VERSION = "facets@1"

const TAG_SYSTEM = `Label each claim with 1-3 facets from this fixed list: ${FACETS.join(", ")}.
Choose what a person would be asking about if this claim were the answer (e.g. "Melanie signed up for a pottery class" -> activity, education).
Return JSON {"tags":{"<id>":["facet",...]}} with every id.`

export async function tagClaims(model: string, claims: Claim[]): Promise<Map<string, Facet[]>> {
  const dir = join(CACHE, "facets")
  mkdirSync(dir, { recursive: true })
  const batches: Claim[][] = []
  for (let i = 0; i < claims.length; i += 60) batches.push(claims.slice(i, i + 60))
  const out = new Map<string, Facet[]>()
  const ok = new Set<string>(FACETS)
  await mapLimit(batches, 3, async (b) => {
    const user = b.map((c) => `${c.id}: ${c.text}`).join("\n")
    const path = join(dir, `${sha1(`${FACET_VERSION}|${model}|${user}`)}.json`)
    let tags: Record<string, string[]>
    if (existsSync(path)) tags = JSON.parse(readFileSync(path, "utf8")) as Record<string, string[]>
    else {
      tags = parseJson<{ tags?: Record<string, string[]> }>(await chat(model, [{ role: "system", content: TAG_SYSTEM }, { role: "user", content: user }], { json: true, maxTokens: 8000 })).tags ?? {}
      writeFileSync(path, JSON.stringify(tags))
    }
    for (const c of b) out.set(c.id, (tags[c.id] ?? []).filter((f): f is Facet => ok.has(f)))
  })
  return out
}

export interface QueryPlan { entities: string[]; facets: Facet[]; aggregate: boolean }

const PLAN_SYSTEM = `You plan a lookup in a memory of claims about people in a conversation between {speakers}.
Given a question, return JSON {"entities":[...],"facets":[...],"aggregate":true|false}
- entities: the people/things the answer must be ABOUT (full names as in the question; resolve "her"/"they" only if obvious).
- facets: 1-3 from: ${FACETS.join(", ")}.
- aggregate: true if the answer is a list or needs several separate facts ("what activities", "how many", "both", "which ... has X done").`

export async function planQuery(model: string, question: string, speakers: string[]): Promise<QueryPlan> {
  const dir = join(CACHE, "plans")
  mkdirSync(dir, { recursive: true })
  const sys = PLAN_SYSTEM.replace("{speakers}", speakers.join(" and "))
  const path = join(dir, `${sha1(`${FACET_VERSION}|${model}|${sys}|${question}`)}.json`)
  if (existsSync(path)) return JSON.parse(readFileSync(path, "utf8")) as QueryPlan
  const raw = parseJson<Partial<QueryPlan>>(await chat(model, [{ role: "system", content: sys }, { role: "user", content: question }], { json: true, maxTokens: 300 }))
  const ok = new Set<string>(FACETS)
  const plan: QueryPlan = { entities: (raw.entities ?? []).map(String), facets: (raw.facets ?? []).filter((f): f is Facet => ok.has(f)), aggregate: !!raw.aggregate }
  writeFileSync(path, JSON.stringify(plan))
  return plan
}
