# RAG Knowledge Assistant — Local LLM + Qdrant

An original, small retrieval-augmented generation (RAG) application for asking questions over local PDF and text documents. It uses Ollama for local embeddings and chat generation, and Qdrant as the vector store.

## Architecture

`PDF/TXT files → text extraction → chunking → Ollama embeddings → Qdrant`

`Question → retrieve relevant chunks → Ollama chat model → grounded answer`

## Stack

- Python
- LangChain community integrations
- Ollama (local embedding and chat models)
- Qdrant (vector database)
- Docker Compose

## Quick start

1. Install Docker Desktop and Python 3.11+.
2. Start Qdrant:
   ```bash
   docker compose up -d
   ```
3. Install Ollama and pull models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
4. Create and activate a virtual environment, then install:
   ```bash
   python -m venv .venv
   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. Put your own non-sensitive PDF or TXT files in `documents/`.
6. Index documents:
   ```bash
   python src/ingest.py
   ```
7. Ask a question:
   ```bash
   python src/ask.py "Summarise the main policy requirements."
   ```

## Configuration

Set environment variables if your local services use non-default endpoints:

- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `QDRANT_URL` (default `http://localhost:6333`)
- `QDRANT_COLLECTION` (default `knowledge_base`)

## Responsible-use notes

- Use only documents you are authorised to process.
- Do not index confidential company or customer material in a personal portfolio.
- Answers are model-generated and can be wrong. Verify important claims against the retrieved source text.
- This is a local portfolio prototype, not a production security boundary.

## Validation

Test with at least three documents and questions whose answers are present, absent and ambiguous. Record retrieval quality, latency and model versions before publishing benchmark claims.

## Attribution

Original project implementation and documentation by Amanaganti Kranthi Kumar. Dependencies and underlying models remain the property of their respective authors and are used under their respective licences.