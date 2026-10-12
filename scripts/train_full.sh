#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"
CELEBA="${1:-../../datasets/celeba/extracted/celeba}"
set +e
bash scripts/check_dataset_size.sh "$CELEBA"
rc=$?
set -e
if [[ "$rc" -eq 3 ]]; then echo "Dataset >= 5 GiB — skip full train." >&2; exit 3; fi
if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi
mkdir -p outputs/logs outputs/checkpoints/full
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
LOG=outputs/logs/train_full.log
echo "Starting FULL Att2MFace CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "$PY" -u -m att2mface.train --config configs/full.yaml "${@:2}" >>"$LOG" 2>&1 &
echo $! >outputs/logs/train_full.pid
echo "PID=$(cat outputs/logs/train_full.pid) log=$LOG"
