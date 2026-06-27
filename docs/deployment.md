# Deployment

## Local Streamlit

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
ollama pull llama3.1
ollama pull nomic-embed-text
streamlit run app.py
```

## Local API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Docker

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.1
docker compose exec ollama ollama pull nomic-embed-text
```

Streamlit runs at `http://localhost:8501`. Ollama runs inside the compose network at `http://ollama:11434`.

## Free-Tier Deployment Notes

Use local Ollama for the final demo when possible. For hosted demos, Streamlit Community Cloud can run non-local LLM paths, while Render/Railway free tiers can host the FastAPI surface. SQLite and FAISS are file-based and should be mounted as persistent volumes where supported.
