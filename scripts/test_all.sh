#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT_DIR/scripts/test_base.sh"
bash "$ROOT_DIR/scripts/test_visual_pe.sh"
bash "$ROOT_DIR/scripts/test_audio_pe.sh"
bash "$ROOT_DIR/scripts/test_audio_visual_pe.sh"
