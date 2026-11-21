"""Load local PDF/TXT files, chunk them, embed with Ollama and store in Qdrant."""
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

DOCS_DIR = Path("documents")
COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def load_documents():
    documents = []
    for path in sorted(DOCS_DIR.glob("**/*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
        elif suffix == ".txt":
            documents.extend(TextLoader(str(path), encoding="utf-8").load())
    return documents


def main():
    source_docs = load_documents()
    if not source_docs:
        raise SystemExit("No PDF/TXT documents found in ./documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150, add_start_index=True
    )
    chunks = splitter.split_documents(source_docs)
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL, base_url=OLLAMA_URL
    )

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION,
        force_recreate=True,
    )
    print(f"Indexed {len(source_docs)} source documents into {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
