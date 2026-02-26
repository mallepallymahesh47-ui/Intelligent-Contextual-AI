import streamlit as st
import os
import tempfile
import shutil
from ingestion import build_corpus
from embedding_vectorstore import build_vectorstore
from retriever_chain import rag_chat


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intelligent Contextual AI",
    layout="wide"
)

# ---------------- ROYAL DARK THEME ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}
.header-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #1e293b;
    border: 1px solid #334155;
    margin-bottom: 20px;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
.stButton>button {
    background-color: #1e40af;
    color: white;
    border-radius: 6px;
    border: none;
}
.stButton>button:hover {
    background-color: #2563eb;
}
.source-box {
    background-color: #1e293b;
    padding: 8px;
    border-radius: 6px;
    border-left: 4px solid #3b82f6;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "file_count" not in st.session_state:
    st.session_state.file_count = 0

if "conversations" not in st.session_state:
    st.session_state.conversations = []


# ---------------- HEADER ----------------
st.markdown("""
<div class="header-box">
    <h2>Intelligent Contextual AI</h2>
    <p>Professional AI Assistant with RAG Capabilities</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:

    # -------- New Chat Button --------
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []

    st.divider()

    # -------- Upload Section --------
    st.subheader("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX, XLSX",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "xlsx", "xls"]
    )

    if st.button("Process Documents", use_container_width=True) and uploaded_files:
        with st.spinner("Processing documents..."):
            try:
                temp_dir = tempfile.mkdtemp()

                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                corpus = build_corpus(temp_dir)
                build_vectorstore(corpus)

                st.session_state.vectorstore = True
                st.session_state.file_count = len(uploaded_files)

                shutil.rmtree(temp_dir)

                st.success("Documents indexed successfully.")

            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # -------- Status --------
    st.subheader("System Status")

    if st.session_state.vectorstore:
        st.success("RAG Mode Active")
    else:
        st.info("Chat Mode Active")

    st.write(f"Files Indexed: {st.session_state.file_count}")

    st.divider()

    # -------- Chat History --------
    st.subheader("Chat History")

    if st.session_state.conversations:
        for idx, conv in enumerate(st.session_state.conversations):
            if st.button(conv["title"], key=f"hist_{idx}"):
                st.session_state.messages = conv["messages"]
    else:
        st.caption("No saved conversations.")


# ---------------- CHAT DISPLAY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- CHAT INPUT ----------------
prompt = st.chat_input("Ask anything...")

if prompt:

    # If starting new conversation, save title
    if not st.session_state.messages:
        title = prompt[:50]
        st.session_state.conversations.append({
            "title": title,
            "messages": []
        })

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                memory = {"messages": st.session_state.messages}

                answer, sources = rag_chat(
                    prompt,
                    st.session_state.vectorstore,
                    memory
                )

                st.markdown(answer)

                # Show source if available
                if sources:
                    st.markdown('<div class="source-box"><b>Source:</b></div>', unsafe_allow_html=True)
                    for src in sources:
                        st.markdown(f"- {src}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                # Update conversation record
                if st.session_state.conversations:
                    st.session_state.conversations[-1]["messages"] = st.session_state.messages.copy()

            except Exception as e:
                st.error(f"Error generating response: {e}")


st.markdown(
    "<hr><center><small>Powered by Qdrant Cloud + Google Gemini</small></center>",
    unsafe_allow_html=True
)