#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/speech-representation-learning/wav2vec2-ser

python -m venv .venv

.venv/bin/python -m pip install --upgrade pip

.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0 \
  torchaudio==2.8.0

.venv/bin/python -m pip install -r requirements.txt
