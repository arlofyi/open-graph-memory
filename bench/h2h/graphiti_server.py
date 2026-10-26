"""E7 adapter: Graphiti (getzep/graphiti, the open-source engine behind Zep)
behind the REST surface that Mem0's own benchmark runner speaks.

- add: one Graphiti episode per runner add (the runner sends one chat message
  per add), `source=EpisodeType.message`, `reference_time` = the session time,
  `group_id` = the runner's user_id. Graphiti then does its own entity and
  fact extraction, deduplication and temporal edge invalidation.
- search, H2H_GRAPHITI_SEARCH=basic (default): Graphiti's out-of-the-box
  `search()` (hybrid BM25 + vector over facts, RRF), scoped to the group. A
  result is one fact (an entity edge), written the way Zep's own guides present
  facts to an LLM: the fact plus its validity range. `created_at` is the fact's
  `valid_at`, else its source episode's time.
- search, H2H_GRAPHITI_SEARCH=combined: Graphiti's advanced `search_()` with
  its `COMBINED_HYBRID_SEARCH_RRF` recipe (facts, entity summaries, episodes
  and communities, each ranked by RRF). Its default recipe needs a
  cross-encoder (logprobs), which the shared chat endpoint could not serve. The four
  ranked lists are interleaved round-robin, facts first.

Models: the LLM is `claude-haiku-5-5` at H2H_CHAT_BASE
(OpenAI-compatible, json_schema structured output), embeddings are local
bge-m3 through Ollama's OpenAI-compatible endpoint, the same models every E7
system gets. The reranker is never called by `search()`.

    NEO4J_URI=bolt://127.0.0.1:7687 NEO4J_PASSWORD=… \
      python -m uvicorn graphiti_server:app --app-dir bench/h2h --port 8912
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("graphiti-h2h")

CHAT_BASE = os.environ["H2H_CHAT_BASE"]
EMBED_BASE = os.getenv("H2H_EMBED_BASE", "http://127.0.0.1:11434/v1")
MODEL = os.getenv("H2H_MODEL", "claude-haiku-5-5")
SEARCH = os.getenv("H2H_GRAPHITI_SEARCH", "basic")
assert SEARCH in ("basic", "combined"), SEARCH

llm_cfg = LLMConfig(api_key=os.getenv("H2H_CHAT_KEY", "local"), model=MODEL, small_model=MODEL, base_url=CHAT_BASE)
graphiti = Graphiti(
    os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    os.getenv("NEO4J_USER", "neo4j"),
    os.environ["NEO4J_PASSWORD"],
    llm_client=OpenAIGenericClient(config=llm_cfg),
    embedder=OpenAIEmbedder(OpenAIEmbedderConfig(api_key="ollama", embedding_model="bge-m3", embedding_dim=1024, base_url=EMBED_BASE)),
    cross_encoder=OpenAIRerankerClient(config=llm_cfg),
)
# Episodes of one group must be added in order; the runner already sends them
# one at a time per conversation, and this lock makes that a guarantee.
locks: dict[str, asyncio.Lock] = {}
episode_time: dict[str, datetime] = {}
counter: dict[str, int] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    await graphiti.build_indices_and_constraints()
    yield
    await graphiti.close()


app = FastAPI(lifespan=lifespan)


class AddRequest(BaseModel):
    messages: list[dict[str, Any]]
    user_id: str
    timestamp: int | None = None


class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = Field(default=100, ge=1, le=1000)
    rerank: bool = False


def _day(d: datetime | None) -> str:
    return d.strftime("%d %B %Y") if d else "present"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "backend": "graphiti", "search": SEARCH, "version": __import__("importlib.metadata").metadata.version("graphiti-core")}


@app.post("/memories")
async def add(req: AddRequest) -> dict[str, Any]:
    body = "\n".join(str(m.get("content", "")).strip() for m in req.messages if str(m.get("content", "")).strip())
    if not body:
        return {"results": []}
    when = datetime.fromtimestamp(req.timestamp, timezone.utc) if req.timestamp is not None else datetime.now(timezone.utc)
    lock = locks.setdefault(req.user_id, asyncio.Lock())
    async with lock:
        n = counter[req.user_id] = counter.get(req.user_id, 0) + 1
        try:
            res = await graphiti.add_episode(
                name=f"{req.user_id}-{n}",
                episode_body=body,
                source_description="chat message",
                reference_time=when,
                source=EpisodeType.message,
                group_id=req.user_id,
            )
        except Exception as e:
            log.exception("add_episode failed")
            raise HTTPException(500, str(e)[:300])
    episode_time[res.episode.uuid] = when
    return {"results": [{"id": e.uuid, "memory": e.fact, "event": "ADD"} for e in res.edges]}


def _edge(e: Any) -> dict[str, Any]:
    when = e.valid_at or next((episode_time[u] for u in e.episodes if u in episode_time), None) or e.created_at
    text = f"{e.fact} (valid from {_day(e.valid_at)} to {_day(e.invalid_at)})" if e.valid_at else e.fact
    return {"id": e.uuid, "memory": text, "created_at": when.isoformat(), "updated_at": when.isoformat()}


@app.post("/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    try:
        if SEARCH == "basic":
            items = [_edge(e) for e in await graphiti.search(req.query, group_ids=[req.user_id], num_results=req.limit)]
        else:
            cfg = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            cfg.limit = req.limit
            r = await graphiti.search_(req.query, config=cfg, group_ids=[req.user_id])
            lists = [
                [_edge(e) for e in r.edges],
                [{"id": nd.uuid, "memory": f"{nd.name}: {nd.summary}"} for nd in r.nodes if nd.summary],
                [{"id": ep.uuid, "memory": ep.content, "created_at": ep.valid_at.isoformat(), "updated_at": ep.valid_at.isoformat()} for ep in r.episodes],
                [{"id": c.uuid, "memory": f"{c.name}: {c.summary}"} for c in r.communities if c.summary],
            ]
            items = []
            for i in range(max((len(x) for x in lists), default=0)):
                items.extend(x[i] for x in lists if i < len(x))
            items = items[: req.limit]
    except Exception as e:
        log.exception("search failed")
        raise HTTPException(500, str(e)[:300])
    n = len(items)
    for i, it in enumerate(items):
        it["score"] = (n - i) / n
    return {"results": items}


@app.delete("/memories")
async def delete_memories(user_id: str) -> dict[str, str]:
    await graphiti.driver.execute_query("MATCH (n {group_id: $g}) DETACH DELETE n", g=user_id)
    return {"message": "deleted"}
