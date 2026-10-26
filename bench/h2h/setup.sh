#!/usr/bin/env bash
# One-time setup for the head-to-head benchmarks (E7, E8). Needs git, uv
# (https://docs.astral.sh/uv/), Docker and a local Ollama with bge-m3.
#
#   bench/h2h/setup.sh
#
# Clones Mem0's benchmark harness at the pinned commit, makes one Python
# environment per system under .cache/h2h/venvs/, and starts Qdrant (for Mem0)
# and Neo4j (for Graphiti) in Docker, bound to localhost.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H="$ROOT/.cache/h2h"
SHA=4b61c5d31b9c668a12b4f5e78064248a02c82d2b
mkdir -p "$H/venvs"

[ -d "$H/memory-benchmarks" ] || git clone -q https://github.com/mem0ai/memory-benchmarks.git "$H/memory-benchmarks"
git -C "$H/memory-benchmarks" checkout -q "$SHA"
[ -x "$H/memory-benchmarks/.venv/bin/python" ] || uv venv -q --python 3.12 "$H/memory-benchmarks/.venv"
uv pip install -q --python "$H/memory-benchmarks/.venv/bin/python" -r "$H/memory-benchmarks/requirements.txt"

for v in mem0 graphiti cognee; do
  [ -x "$H/venvs/$v/bin/python" ] || uv venv -q --python 3.12 "$H/venvs/$v"
  uv pip install -q --python "$H/venvs/$v/bin/python" -r "$ROOT/bench/h2h/requirements-$v.txt"
done

docker start ogm-h2h-qdrant >/dev/null 2>&1 || docker run -d --name ogm-h2h-qdrant -p 127.0.0.1:6333:6333 qdrant/qdrant >/dev/null
docker start ogm-h2h-neo4j >/dev/null 2>&1 || docker run -d --name ogm-h2h-neo4j -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
  -e NEO4J_AUTH=neo4j/h2hpassword neo4j:5.26 >/dev/null
ollama pull bge-m3 >/dev/null
echo "setup done"
