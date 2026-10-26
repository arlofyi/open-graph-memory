#!/usr/bin/env bash
# E7 runner: Mem0's own LoCoMo harness, unmodified, against one system's
# Mem0-REST endpoint.
#
#   bench/h2h/run.sh <name> <port> <dev|test> [extra harness args]
#
# The harness is a clean clone of mem0ai/memory-benchmarks in .cache/h2h/
# (bench/h2h/setup.sh), pinned at 4b61c5d. This script refuses to run if any
# tracked file differs from that commit. Answerer and judge are
# claude-haiku-5-5 at H2H_CHAT_BASE (or H2H_CHAT_BASE_QA). The dataset is
# $LOCOMO_PATH, default .cache/locomo/locomo10.json. Results go to
# .cache/h2h/<split>/ (gitignored; LoCoMo is CC BY-NC and the predictions
# quote it).

set -euo pipefail
NAME="$1"; PORT="$2"; SPLIT="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="${MEM0_BENCH_DIR:-$ROOT/.cache/h2h/memory-benchmarks}"
SHA=4b61c5d31b9c668a12b4f5e78064248a02c82d2b
DATA="${LOCOMO_PATH:-$ROOT/.cache/locomo/locomo10.json}"

[ "$(git -C "$BENCH" rev-parse HEAD)" = "$SHA" ] || { echo "harness is not at $SHA" >&2; exit 1; }
git -C "$BENCH" diff --quiet HEAD -- || { echo "harness has local modifications; refusing" >&2; exit 1; }
case "$SPLIT" in
  dev) CONVS=0,1 ;;
  test) CONVS=2,3,4,5,6,7,8,9 ;;
  *) echo "split must be dev or test" >&2; exit 2 ;;
esac
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "nothing healthy on :$PORT" >&2; exit 1; }
# The harness loads its .env with override=True, so the endpoint goes there.
QA_BASE="${H2H_CHAT_BASE_QA:-${H2H_CHAT_BASE:?set H2H_CHAT_BASE to an OpenAI-compatible endpoint}}"
printf 'MEM0_BACKEND=oss\nOPENAI_BASE_URL=%s\nOPENAI_API_KEY=%s\n' "$QA_BASE" "${H2H_CHAT_KEY:-local}" > "$BENCH/.env"

OUT="$ROOT/.cache/h2h/$SPLIT"
mkdir -p "$OUT"
printf '%s\tname=%s\tport=%s\tsplit=%s\thealth=%s\n' "$(date -u +%FT%TZ)" "$NAME" "$PORT" "$SPLIT" "$(curl -s "http://127.0.0.1:$PORT/health")" >> "$OUT/RUNS.log"
cd "$BENCH"
.venv/bin/python -m benchmarks.locomo.run \
    --project-name "$NAME" --run-id "${H2H_RUN_ID:-$NAME}" --backend oss --mem0-host "http://127.0.0.1:$PORT" \
    --answerer-model claude-haiku-5-5 --judge-model claude-haiku-5-5 \
    --top-k 200 --top-k-cutoffs 10,50,200 \
    --conversations "$CONVS" --dataset-path "$DATA" \
    --output-dir "$OUT" --max-workers "${H2H_WORKERS:-4}" "${@:4}"
