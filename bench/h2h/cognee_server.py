"""E7 adapter: Cognee (topoteretes/cognee) behind the REST surface that Mem0's
own benchmark runner speaks.

Cognee is document-oriented: `add` then `cognify` (chunk, extract a graph,
summarise, embed). The runner sends one chat message per add, and cognifying
after every message would rebuild constantly, so this adapter does what the
OGM adapter does: messages are grouped into one document per session (all adds
with the same `timestamp`), headed with the session date, and the dataset is
cognified lazily on the first search after new adds. Nothing about the
questions is visible at that point.

Search is Cognee's default (`HYBRID_COMPLETION`) with `only_context=True`, so
Cognee retrieves and formats its context but does not answer; the runner's
answerer does. That context is one sectioned block ("Relevant passages",
"Relevant entities"); it is split into one result per passage and per entity
fact, in Cognee's order, deduplicated. A passage's `created_at` is its
session's date; entity facts carry none.

Models: Cognee's LLM is `claude-haiku-5-5` at the shared chat endpoint (litellm,
instructor `json_schema_mode`), embeddings are local bge-m3 through Ollama:
the same models every E7 system gets.

    . bench/h2h/cognee.env && python -m uvicorn cognee_server:app --app-dir bench/h2h --port 8914
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import cognee
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cognee-h2h")

ROOT = os.path.abspath(os.getenv("H2H_COGNEE_ROOT", ".cache/h2h/cognee"))
cognee.config.system_root_directory(os.path.join(ROOT, "system"))
cognee.config.data_root_directory(os.path.join(ROOT, "data"))
app = FastAPI()

# user_id → {timestamp: [message, …]}; which sessions are already cognified
sessions: dict[str, dict[int, list[str]]] = {}
added: dict[str, set[int]] = {}
dirty: set[str] = set()
build_lock = asyncio.Lock()  # Cognee keeps global state; one cognify at a time


class AddRequest(BaseModel):
    messages: list[dict[str, Any]]
    user_id: str
    timestamp: int | None = None


class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = Field(default=100, ge=1, le=1000)
    rerank: bool = False


def _dataset(user_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", user_id)


def _doc(ts: int, lines: list[str]) -> str:
    d = datetime.fromtimestamp(ts, timezone.utc)
    return f"Conversation on {d.strftime('%A, %d %B %Y, %H:%M')} UTC.\n" + "\n".join(lines)


async def _build(user_id: str) -> None:
    async with build_lock:
        if user_id not in dirty:
            return
        done = added.setdefault(user_id, set())
        new = [ts for ts in sorted(sessions[user_id]) if ts not in done]
        t0 = datetime.now()
        for ts in new:
            await cognee.add(_doc(ts, sessions[user_id][ts]), dataset_name=_dataset(user_id))
            done.add(ts)
        # Stay dirty until cognify succeeds: a failed build (an LLM error, a
        # spend cap) must be retried on the next search, not leave the
        # dataset's graph empty for good.
        await cognee.cognify(datasets=[_dataset(user_id)])
        dirty.discard(user_id)
        log.info("cognified %s: %d new sessions in %.0f s", user_id, len(new), (datetime.now() - t0).total_seconds())


def _split(context: str) -> list[tuple[str, str]]:
    """Cognee's sectioned context → [(kind, text)] in its order."""
    body = context.split("Context:\n", 1)[-1].strip().strip("`")
    out: list[tuple[str, str]] = []
    # Split only on Cognee's own section headers and passage separator: the
    # stored text itself can hold blank lines and markdown "## " headings
    # (LongMemEval's assistant replies do), and splitting on those cut one
    # passage into many fragments.
    for section in re.split(r"^## (?=(?:Relevant passages|Relevant entities|Related facts|Global context|Fact conflicts)\s*$)", body, flags=re.M):
        if section.startswith("Relevant passages"):
            for p in re.split(r"\n-{3,}\n", section.split("\n", 1)[-1]):
                if p.strip():
                    out.append(("passage", p.strip()))
        elif section.strip():
            for line in section.splitlines():
                if line.startswith("- "):
                    out.append(("fact", line[2:].strip()))
    return out


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "backend": "cognee", "version": __import__("importlib.metadata").metadata.version("cognee")}


@app.post("/memories")
def add(req: AddRequest) -> dict[str, Any]:
    ts = req.timestamp if req.timestamp is not None else int(datetime.now(timezone.utc).timestamp())
    lines = [str(m.get("content", "")).strip() for m in req.messages if str(m.get("content", "")).strip()]
    sessions.setdefault(req.user_id, {}).setdefault(ts, []).extend(lines)
    dirty.add(req.user_id)
    return {"results": [{"id": f"{req.user_id}-{ts}-{i}", "memory": t, "event": "ADD"} for i, t in enumerate(lines)]}


@app.post("/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    if req.user_id not in sessions:
        return {"results": []}
    try:
        await _build(req.user_id)
        res = await cognee.search(req.query, datasets=[_dataset(req.user_id)], only_context=True, top_k=req.limit)
    except Exception as e:
        log.exception("search failed")
        raise HTTPException(500, str(e)[:300])
    context = "\n".join(str(r.get("search_result", "")) if isinstance(r, dict) else str(r) for r in res)
    docs = {ts: _doc(ts, lines) for ts, lines in sessions[req.user_id].items()}
    seen: set[str] = set()
    items = []
    for kind, text in _split(context):
        if text in seen:
            continue
        seen.add(text)
        item: dict[str, Any] = {"id": f"{kind}-{len(items)}", "memory": text, "metadata": {"cognee_kind": kind}}
        if kind == "passage":
            ts = next((t for t, d in docs.items() if text[:120] in d), None)
            if ts is not None:
                item["created_at"] = item["updated_at"] = datetime.fromtimestamp(ts, timezone.utc).isoformat()
        items.append(item)
    items = items[: req.limit]
    n = len(items)
    for i, it in enumerate(items):
        it["score"] = (n - i) / n
    return {"results": items}


@app.delete("/memories")
async def delete_memories(user_id: str) -> dict[str, str]:
    sessions.pop(user_id, None)
    added.pop(user_id, None)
    return {"message": "deleted"}
