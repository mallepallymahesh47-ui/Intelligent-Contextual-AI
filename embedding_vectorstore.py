from dotenv import load_dotenv
load_dotenv()

import os
import uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

if not LANGCHAIN_API_KEY:
    raise ValueError("LANGCHAIN_API_KEY is not set")

if not LANGCHAIN_PROJECT:
    raise ValueError("LANGCHAIN_PROJECT is not set")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "multimodal_rag"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def build_vectorstore(corpus_with_sources):

    # 🔥 Always recreate collection
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    embedding_size = len(embedding_model.embed_query("dimension test"))

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_size,
            distance=Distance.COSINE,
        ),
    )

    points = []

    for file_name, text in corpus_with_sources:

        if not text or not text.strip():
            continue

        chunks = chunk_text(text)

        for chunk in chunks:
            vector = embedding_model.embed_query(chunk)

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": file_name
                    }
                )
            )

    if not points:
        print("No valid content found.")
        return False

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Indexed {len(points)} chunks successfully.")

    return True