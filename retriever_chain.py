from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "multimodal_rag"

llm = ChatGoogleGenerativeAI(
    model="gemma-3-1b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# ---------------- RETRIEVE CONTEXT ----------------
def retrieve_context(query, k=5):

    query_vector = embedding_model.embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k
    )

    texts = []
    sources = set()
    max_score = 0.0

    for point in results.points:
        score = point.score or 0.0
        max_score = max(max_score, score)

        payload = point.payload or {}

        if "text" in payload:
            texts.append(payload["text"])

        if "source" in payload:
            sources.add(payload["source"])

    return "\n".join(texts), list(sources), max_score


# ---------------- MAIN CHAT ----------------
def rag_chat(query, vectorstore, memory):

    # No documents indexed
    if not vectorstore:
        answer = llm.invoke(query).content
        return answer, None

    context, sources, max_score = retrieve_context(query)

    # If no meaningful similarity → normal chat
    if max_score < 0.30:
        answer = llm.invoke(query).content
        return answer, None

    if not context.strip():
        answer = llm.invoke(query).content
        return answer, None

    # 🔥 STRONG CONTROL PROMPT
    prompt = f"""
You are answering based ONLY on the uploaded document.

You MUST:
- Use the context below.
- NEVER ask the user to provide the document.
- NEVER say "please provide the file".
- NEVER say "paste the document".
- If partial information exists, summarize what is available.
- Do not mention the word "context" in your answer.

Document Content:
{context}

User Question:
{query}

Provide a complete and professional answer based strictly on the document.
"""

    answer = llm.invoke(prompt).content

    # 🚫 Safety filter: if model still asks for file, override
    forbidden_phrases = [
        "please provide",
        "paste the document",
        "i need the text",
        "provide the file"
    ]

    if any(p.lower() in answer.lower() for p in forbidden_phrases):
        answer = llm.invoke(
            f"Summarize this document content clearly:\n{context}"
        ).content

    return answer, sources