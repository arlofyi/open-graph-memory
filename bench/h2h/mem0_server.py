"""E7 adapter: Mem0 OSS (the `mem0ai` package) behind the REST surface that
Mem0's own benchmark runner speaks.

Why not the runner's own `docker/mem0` server? It was written against a
`mem0ai` branch that no longer exists. Against the released package (2.2.1,
which contains that branch's "V3 phased batch pipeline") its /search passes
`user_id` where `filters` is now required, and its /memories drops the
session `timestamp`, because `Memory.add(timestamp=…)` is Platform-only. Run
as is, Mem0 would date every memory "today" and could not answer any temporal
question. That would be a straw man, so this adapter gives Mem0 the date the
way the package supports it:

- add: `prompt` (Mem0's custom-instructions hook, appended to its extraction
  prompt) says when the conversation took place, and the session time is
  stored as metadata `session_ts`;
- search: `filters={"user_id": …}`, and `created_at` is reported as that
  session time, as the runner expects (it prints `created_at` as the memory's
  date) and as Mem0 Platform does with `timestamp`.

Everything else is Mem0's own default behaviour.

    MEM0_CONFIG_PATH=.cache/h2h/mem0-config.yaml QDRANT_HOST=127.0.0.1 \
      python -m uvicorn mem0_server:app --app-dir bench/h2h --port 8911
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from mem0 import Memory
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mem0-h2h")

cfg = yaml.safe_load(open(os.environ["MEM0_CONFIG_PATH"]))
cfg.pop("version", None)
cfg.setdefault("vector_store", {
    "provider": "qdrant",
    "config": {
        "host": os.getenv("QDRANT_HOST", "127.0.0.1"),
        "port": int(os.getenv("QDRANT_PORT", "6333")),
        "collection_name": os.getenv("COLLECTION_NAME", "h2h"),
        "embedding_model_dims": cfg.get("embedder", {}).get("config", {}).get("embedding_dims", 1024),
    },
})
memory = Memory.from_config(cfg)
app = FastAPI()


class AddRequest(BaseModel):
    messages: list[dict[str, Any]]
    user_id: str
    timestamp: int | None = None
    metadata: dict[str, Any] | None = None
    custom_instructions: str | None = None


class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = Field(default=100, ge=1, le=1000)
    rerank: bool = False


def _date_line(ts: int) -> str:
    d = datetime.fromtimestamp(ts, timezone.utc)
    return f"These messages are from a conversation that took place on {d.strftime('%A, %d %B %Y, %H:%M')} UTC. Resolve relative times (yesterday, last week, next month) against that date, not today's date."


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "backend": "mem0-oss", "version": __import__("importlib.metadata").metadata.version("mem0ai")}


@app.post("/memories")
def add(req: AddRequest) -> dict[str, Any]:
    meta = dict(req.metadata or {})
    prompt = req.custom_instructions
    if req.timestamp is not None:
        meta["session_ts"] = req.timestamp
        prompt = f"{prompt}\n{_date_line(req.timestamp)}" if prompt else _date_line(req.timestamp)
    try:
        out = memory.add(req.messages, user_id=req.user_id, metadata=meta, prompt=prompt)
    except Exception as e:
        log.exception("add failed")
        raise HTTPException(500, str(e)[:300])
    return out if isinstance(out, dict) else {"results": out}


@app.post("/search")
def search(req: SearchRequest) -> dict[str, Any]:
    try:
        out = memory.search(req.query, top_k=req.limit, filters={"user_id": req.user_id}, rerank=req.rerank)
    except Exception as e:
        log.exception("search failed")
        raise HTTPException(500, str(e)[:300])
    results = out.get("results", []) if isinstance(out, dict) else out
    for r in results:
        ts = (r.get("metadata") or {}).get("session_ts")
        if ts is not None:
            when = datetime.fromtimestamp(int(ts), timezone.utc).isoformat()
            r["created_at"] = when
            r["updated_at"] = when
    return {"results": results}


@app.get("/memories")
def list_memories(user_id: str) -> dict[str, Any]:
    out = memory.get_all(filters={"user_id": user_id}, top_k=1000)
    return out if isinstance(out, dict) else {"results": out}


@app.delete("/memories")
def delete_memories(user_id: str) -> dict[str, str]:
    memory.delete_all(user_id=user_id)
    return {"message": "deleted"}
