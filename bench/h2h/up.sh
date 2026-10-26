#!/usr/bin/env bash
# Start the head-to-head servers (E7, E8): every public arm behind the Mem0
# OSS REST surface that Mem0's benchmark runner speaks. Run setup.sh first.
#
#   H2H_CHAT_BASE=http://127.0.0.1:4000/v1 [H2H_CHAT_KEY=…] bench/h2h/up.sh [store name]
#
# H2H_CHAT_BASE is any OpenAI-compatible chat-completions endpoint serving
# claude-haiku-5-5 (for example a LiteLLM proxy in front of Anthropic's API).
# To measure each system's ingest spend separately, give each one its own
# endpoint with H2H_CHAT_BASE_<ARM> (ARM = MEM0, GRAPHITI, COGNEE, DENSE,
# DENSEX, R2A); each falls back to H2H_CHAT_BASE. Embeddings are local bge-m3.
#
# Ports: mem0 8911, graphiti 8912, graphiti-combined 8913 (searches the same
# graph as 8912), cognee 8914, ogm-dense 8915, ogm-denseX 8916, ogm-r2a 8917.
# The arlo-memory arm in docs/RESULTS.md is not public and is not started here.
# Logs go to .cache/h2h/logs/.

set -euo pipefail
: "${H2H_CHAT_BASE:?set H2H_CHAT_BASE to an OpenAI-compatible endpoint}"
KEY="${H2H_CHAT_KEY:-local}"
NAME="${1:-h2h}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H="$ROOT/.cache/h2h"; LOGS="$H/logs"; mkdir -p "$LOGS"
base() { local v="H2H_CHAT_BASE_$1"; echo "${!v:-$H2H_CHAT_BASE}"; }
cd "$ROOT"

# Mem0: its own Qdrant collection and history database.
sed -e "s#\${H2H_CHAT_BASE}#$(base MEM0)#" -e "s#\${H2H_CHAT_KEY}#$KEY#" -e "s#\${H2H_DIR}#$H#" \
  -e "s#mem0-history.db#mem0-history-$NAME.db#" bench/h2h/mem0-config.example.yaml > "$H/mem0-config-$NAME.yaml"
MEM0_CONFIG_PATH="$H/mem0-config-$NAME.yaml" QDRANT_HOST=127.0.0.1 COLLECTION_NAME="$NAME" \
  nohup "$H/venvs/mem0/bin/python" -m uvicorn mem0_server:app --app-dir bench/h2h --host 127.0.0.1 --port 8911 >> "$LOGS/mem0.log" 2>&1 < /dev/null &
# Graphiti: one ingest (basic server); the combined server only searches the same groups.
for spec in basic:8912 combined:8913; do
  NEO4J_PASSWORD=h2hpassword H2H_CHAT_BASE="$(base GRAPHITI)" H2H_CHAT_KEY="$KEY" H2H_GRAPHITI_SEARCH="${spec%%:*}" \
    nohup "$H/venvs/graphiti/bin/python" -m uvicorn graphiti_server:app --app-dir bench/h2h --host 127.0.0.1 --port "${spec##*:}" >> "$LOGS/graphiti-${spec%%:*}.log" 2>&1 < /dev/null &
done
# Cognee: its own root directory.
( set -a; . bench/h2h/cognee.env; set +a
  LLM_ENDPOINT="$(base COGNEE)" LLM_API_KEY="$KEY" H2H_COGNEE_ROOT="$H/cognee-$NAME" \
  nohup "$H/venvs/cognee/bin/python" -m uvicorn cognee_server:app --app-dir bench/h2h --host 127.0.0.1 --port 8914 >> "$LOGS/cognee.log" 2>&1 < /dev/null & )
# open-graph-memory: three read paths.
for spec in dense:8915:DENSE denseX:8916:DENSEX r2a:8917:R2A; do
  IFS=: read -r mode port arm <<< "$spec"
  OGM_EXTRACTOR=claims@2 OGM_H2H_MODE="$mode" OGM_CHAT_BASE="$(base "$arm")" OGM_CHAT_KEY="$KEY" \
    nohup node bench/h2h/ogm-server.ts --port "$port" >> "$LOGS/ogm-$mode.log" 2>&1 < /dev/null &
done

for i in $(seq 1 45); do
  sleep 2; bad=""
  for p in 8911 8912 8913 8914 8915 8916 8917; do curl -sf -m2 "127.0.0.1:$p/health" >/dev/null || bad="$bad $p"; done
  [ -z "$bad" ] && { echo "all 7 servers up"; exit 0; }
done
echo "not healthy:$bad (see $LOGS)" >&2; exit 1
