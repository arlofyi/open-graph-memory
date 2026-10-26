#!/usr/bin/env bash
# E8 runner: the LongMemEval runner of Mem0's own benchmark repo, unmodified,
# against one system's Mem0-REST endpoint.
#
#   bench/h2h/run-lme.sh <name> <port> <dataset file> <out split> [extra harness args]
#
# Same pinned clone and checks as run.sh. Answerer and judge are
# claude-haiku-5-5 at H2H_CHAT_BASE (or H2H_CHAT_BASE_QA). Results go to
# .cache/h2h/e8-<out split>/ (gitignored: the predictions quote the dataset).
# H2H_WORKERS sets the runner's parallelism (default 4); H2H_RUN_ID overrides
# the run id, which names the per-question user ids.

set -euo pipefail
NAME="$1"; PORT="$2"; DATA="$3"; SPLIT="$4"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="${MEM0_BENCH_DIR:-$ROOT/.cache/h2h/memory-benchmarks}"
SHA=4b61c5d31b9c668a12b4f5e78064248a02c82d2b

[ "$(git -C "$BENCH" rev-parse HEAD)" = "$SHA" ] || { echo "harness is not at $SHA" >&2; exit 1; }
git -C "$BENCH" diff --quiet HEAD -- || { echo "harness has local modifications; refusing" >&2; exit 1; }
[ -f "$DATA" ] || { echo "no dataset file $DATA" >&2; exit 1; }
DATA="$(cd "$(dirname "$DATA")" && pwd)/$(basename "$DATA")"
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "nothing healthy on :$PORT" >&2; exit 1; }
# The harness loads its .env with override=True, so the endpoint goes there.
QA_BASE="${H2H_CHAT_BASE_QA:-${H2H_CHAT_BASE:?set H2H_CHAT_BASE to an OpenAI-compatible endpoint}}"
printf 'MEM0_BACKEND=oss\nOPENAI_BASE_URL=%s\nOPENAI_API_KEY=%s\n' "$QA_BASE" "${H2H_CHAT_KEY:-local}" > "$BENCH/.env"

OUT="$ROOT/.cache/h2h/e8-$SPLIT"
mkdir -p "$OUT"
printf '%s\tname=%s\tport=%s\tdata=%s\thealth=%s\n' "$(date -u +%FT%TZ)" "$NAME" "$PORT" "$(basename "$DATA")" "$(curl -s "http://127.0.0.1:$PORT/health")" >> "$OUT/RUNS.log"
cd "$BENCH"
.venv/bin/python -m benchmarks.longmemeval.run \
    --project-name "$NAME" --run-id "${H2H_RUN_ID:-$NAME}" --backend oss --mem0-host "http://127.0.0.1:$PORT" \
    --answerer-model claude-haiku-5-5 --judge-model claude-haiku-5-5 \
    --top-k 200 --top-k-cutoffs 10,50,200 \
    --dataset-path "$DATA" --output-dir "$OUT" --max-workers "${H2H_WORKERS:-4}" "${@:5}"
