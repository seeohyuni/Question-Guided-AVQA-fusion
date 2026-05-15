#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
NUM_WORKERS="${NUM_WORKERS:-8}"

"$PYTHON_BIN" models/audio_visual_pe/main_avst.py --mode test \
  --audio_dir ./features/msclap_all \
  --audio_fallback_dir ./features/msclap_av_counting \
  --video_res14x14_dir ./features/clip_patch14_all \
  --video_res14x14_fallback_dir ./features/clip_patch14_av_counting \
  --text_feat_dir ./features/clip_text_all \
  --text_feat_fallback_dir ./features/clip_text_av_counting/per_question \
  --label_train ./dataset/json/avqa-train.json \
  --label_test ./dataset/json/avqa-test.json \
  --model_save_dir ./checkpoints \
  --checkpoint audio_visual_pe \
  --num_workers "$NUM_WORKERS"
