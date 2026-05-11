import os
from dotenv import load_dotenv
import shutil
import uvicorn
import streamlit as st
import requests
import threading
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, ChatPromptTemplate
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from llama_index.readers.file import PDFReader, DocxReader, FlatReader

# Load environment variables
load_dotenv()

app = FastAPI()

# ============= LLM CONFIG =============
Settings.llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    token=os.getenv("HF_TOKEN"),
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

PERSIST_DIR = "./db"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

class Query(BaseModel):
    question: str

# ============= INGESTION PIPELINE =============

def data_ingestion():
    documents = []
    file_count = 0

    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)

        if not os.path.isfile(file_path):
            continue

        print(f"Processing: {file}")
        docs = []

        try:
            if file.endswith(".pdf"):
                docs = PDFReader().load_data(file_path)
            elif file.endswith(".docx"):
                docs = DocxReader().load_data(file_path)
            elif file.endswith(".txt"):
                docs = FlatReader().load_data(file_path)
            else:
                continue

            for doc in docs:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["filename"] = file

            documents.extend(docs)
            file_count += 1

        except Exception as e:
            print(f"Error: {file} -> {str(e)}")

    if not documents:
        print("No documents found")
        return

    print(f"Loaded {file_count} files")

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    print("Index saved")

# ============= QUERY PIPELINE =============

def handle_query(query):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    except Exception as e:
        return {"answer": "No index found. Upload documents first.", "error": str(e)}

    chat_text_qa_msgs = [
        (
       
        "system",
        """You are MULTI-DOC-AI, an expert document analyst and intelligent assistant.

You have access to:
- Current user question
- Previous conversation history (if provided)
- Uploaded document context

Your behavior:

1. MEMORY AWARENESS:
- Use previous questions and answers to understand context.
- If the current question depends on earlier conversation, refer to it.
- Do NOT ask the user to repeat information already mentioned earlier.

2. DOCUMENT RULE:
- Answer ONLY using uploaded documents.
- You may combine document context + conversation context to improve clarity.

3. INTELLIGENCE RULE:
- Understand intent, not just exact words.
- If information is not directly stated, infer logically from the document context only.

4. UNKNOWN INFO RULE:
- If answer is not in documents or cannot be inferred, say:
  "This information is not found in the uploaded documents."

5. RESPONSE STYLE:
- Simple, clear, and human-like explanation
- Use bullet points if helpful
- Keep answers structured and easy to read
- Avoid repetition from previous answers unless needed for clarity
"""
  
        )
    ]

    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    query_engine = index.as_query_engine(
        text_qa_template=text_qa_template,
        similarity_top_k=5
    )

    response = query_engine.query(query)
    answer_text = response.response if hasattr(response, "response") else str(response)

    return {"answer": answer_text}

# ============= FASTAPI ENDPOINTS =============

@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):

    print(f"Received {len(files)} files")

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".docx", ".txt"]:
            raise HTTPException(status_code=400, detail="Invalid file type")

    os.makedirs(DATA_DIR, exist_ok=True)

    saved_files = []

    for file in files:
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append(file.filename)

    print("Saved:", saved_files)

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    data_ingestion()

    return {
        "message": f"Uploaded {len(files)} files",
        "files": saved_files
    }

@app.get("/documents")
async def list_documents():
    if not os.path.exists(DATA_DIR):
        return {"documents": []}

    docs = os.listdir(DATA_DIR)
    return {"documents": docs, "count": len(docs)}

@app.post("/query")
async def query_document(query: Query):

    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise HTTPException(status_code=400, detail="No documents uploaded")

    return handle_query(query.question)

@app.delete("/clear")
async def clear_documents():

    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    return {"message": "Cleared"}

# ============= STREAMLIT UI =============

def streamlit_ui():

    st.set_page_config(page_title="Multi-Doc AI", layout="wide")
    st.title("🤖 Multi Document Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload documents and ask questions"}
        ]

    with st.sidebar:
        st.header("Upload Files")

        uploaded_files = st.file_uploader(
            "Upload PDF/DOCX/TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button("Upload") and uploaded_files:
            files = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in uploaded_files
            ]

            res = requests.post("http://localhost:8000/upload", files=files)

            if res.status_code == 200:
                st.success("Uploaded successfully")

        if st.button("Clear All"):
            requests.delete("http://localhost:8000/clear")
            st.success("Cleared")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask something...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        res = requests.post(
            "http://localhost:8000/query",
            json={"question": prompt}
        )

        if res.status_code == 200:
            answer = res.json()["answer"]
        else:
            answer = "Error getting response"

        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()

# ============= RUN APP =============

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    threading.Thread(target=run_fastapi, daemon=True).start()
    streamlit_ui()