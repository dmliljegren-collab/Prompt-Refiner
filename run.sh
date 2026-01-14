#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run: python3 -m venv .venv" >&2
  exit 1
fi

source .venv/bin/activate

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. The app will run but won't call OpenAI." >&2
fi

exec uvicorn main:app --reload
