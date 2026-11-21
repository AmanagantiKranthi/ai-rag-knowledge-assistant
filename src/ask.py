"""Ask a question against the local Qdrant knowledge collection."""
import os
import sys

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using only the supplied context. If the context does not contain "
     "the answer, say that you cannot find it in the indexed documents. "
     "Do not invent citations or facts. Context:\n{context}"),
    ("human", "{question}"),
])


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/ask.py "your question"')

    question = " ".join(sys.argv[1:])
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION,
        url=QDRANT_URL,
    )
    matches = store.similarity_search_with_score(question, k=4)
    if not matches:
        raise SystemExit("No matching document chunks found. Run ingest.py first.")

    context_parts = []
    for doc, score in matches:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = f"{source}" + (f", page {page + 1}" if isinstance(page, int) else "")
        context_parts.append(f"[Source: {location}; similarity score: {score:.3f}]\n{doc.page_content}")

    chain = PROMPT | ChatOllama(
        model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0
    )
    response = chain.invoke({
        "context": "\n\n---\n\n".join(context_parts),
        "question": question,
    })
    print(response.content)
    print("\nRetrieved sources:")
    for doc, score in matches:
        print(f"- {doc.metadata.get('source', 'unknown')} (score={score:.3f})")


if __name__ == "__main__":
    main()
