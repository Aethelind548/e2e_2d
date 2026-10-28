#!/usr/bin/env bash
# Full Route Preview V1 generation from a readable Bench2Drive source.
#
# The historical /data/datasets mount is currently a stack of stale
# emergency_ro/shutdown ext4 mounts (sda1/sdb1/sdc1) and must not be remounted
# blindly.  The live dataset is mounted at:
#   /run/media/aethelind/datasets/Bench2Drive-V0.0.4/v1
# Override SOURCE/TARGET explicitly if the canonical mount is restored.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${ROUTE_PREVIEW_PYTHON:-/home/aethelind/miniconda3/envs/navsim_gpu/bin/python}
SOURCE=${ROUTE_PREVIEW_SOURCE:-/run/media/aethelind/datasets/Bench2Drive-V0.0.4/v1}
TARGET=${ROUTE_PREVIEW_TARGET:-/run/media/aethelind/datasets/Bench2Drive-V0.0.4/route_preview_v1}
COMPACT=${ROUTE_PREVIEW_COMPACT:-$ROOT/outputs/diffusiondrive_v1/route_preview_data_audit/20261022_readonly}
WORKERS=${ROUTE_PREVIEW_WORKERS:-8}
STAMP=$(date '+%Y%m%d_%H%M%S')
EVIDENCE="$ROOT/outputs/diffusiondrive_v1/route_preview_v11/full_generation_${STAMP}"
mkdir -p "$EVIDENCE"
LOG="$EVIDENCE/full_generation.log"

printf 'timestamp=%s\nworkspace=%s\npython=%s\nsource=%s\ntarget=%s\ncompact=%s\nworkers=%s\n' \
  "$STAMP" "$ROOT" "$PY" "$SOURCE" "$TARGET" "$COMPACT" "$WORKERS" | tee "$LOG"
printf '%s\n' '--- capacity ---' | tee -a "$LOG"
df -h "$SOURCE" "$TARGET" 2>&1 | tee -a "$LOG" || true

if [ ! -x "$PY" ]; then
  printf 'ERROR: python not executable: %s\n' "$PY" | tee -a "$LOG"
  exit 2
fi
if [ ! -d "$SOURCE" ]; then
  printf 'ERROR: source directory unavailable: %s\n' "$SOURCE" | tee -a "$LOG"
  printf 'Do not blindly remount /data/datasets; inspect findmnt -T /data/datasets first.\n' | tee -a "$LOG"
  exit 2
fi

# Inventory is read-only and refuses to write target data if the source fails.
"$PY" "$ROOT/tools/validation/b2d/route_preview_inventory.py" \
  --source "$SOURCE" --target "$TARGET" \
  --output "$EVIDENCE/dataset_inventory.json" 2>&1 | tee -a "$LOG"
INV_RC=${PIPESTATUS[0]}
if [ "$INV_RC" -ne 0 ]; then
  printf 'BLOCKED_SOURCE_IO: inventory failed; refusing full generation.\n' | tee -a "$LOG"
  exit 2
fi
if ! "$PY" - "$EVIDENCE/dataset_inventory.json" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["source_scan"]["status"] == "PASS" else 1)
PY
then
  printf 'BLOCKED_SOURCE_IO: inventory did not pass; refusing full generation.\n' | tee -a "$LOG"
  exit 2
fi

printf '%s\n' '--- full generation ---' | tee -a "$LOG"
"$PY" "$ROOT/tools/validation/b2d/route_preview_generate.py" \
  --source-root "$SOURCE" --output-root "$TARGET" \
  --compact-dir "$COMPACT" --workers "$WORKERS" --resume "$@" 2>&1 | tee -a "$LOG"
GEN_RC=${PIPESTATUS[0]}
printf 'generation_exit=%s evidence=%s\n' "$GEN_RC" "$EVIDENCE" | tee -a "$LOG"
exit "$GEN_RC"
