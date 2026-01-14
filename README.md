# Prompt Refinery

A lightweight FastAPI web app that refines prompts or generates new ones based on user input.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## Next steps

- Optionally set `OPENAI_MODEL` (defaults to `gpt-4o-mini`).
- Add prompt history or export options.
