#!/usr/bin/env bash
set -e

cd /workspaces/speech-representation-learning/02-wav2vec2-ser

python -m venv .venv

.venv/bin/python -m pip install --upgrade pip

.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.6.0 \
  torchaudio==2.6.0

.venv/bin/python -m pip install -r requirements.txt
