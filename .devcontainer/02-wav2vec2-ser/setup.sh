#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/speech-representation-learning/02-wav2vec2-ser

python -m venv .venv

.venv/bin/python -m pip install --upgrade pip

.venv/bin/python -m pip install -r requirements.txt
