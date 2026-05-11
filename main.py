import os
import shutil
import uvicorn
from typing import List

from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.app import data_ingestion, handle_query, DATA_DIR

PERSIST_DIR = "./db"

class Query(BaseModel):
    question: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    """
    Upload multiple files at once
    """
    print(f"Received {len(files)} file(s)")
    
    # Validate all files first
    for file in files:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".pdf", ".docx", ".txt"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only PDF, DOCX, and TXT are allowed."
            )
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save all new files
    saved_files = []
    for file in files:
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append(file.filename)
    
    print("Saved files:", saved_files)
    
    # Clear old index
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    
    print("FILES IN DATA_DIR:", os.listdir(DATA_DIR))
    
    # Re-ingest all data
    data_ingestion()
    
    return {
        "message": f"Successfully uploaded and processed {len(files)} file(s)",
        "files": saved_files
    }

@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents
    """
    if not os.path.exists(DATA_DIR):
        return {"documents": []}
    
    docs = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    return {"documents": docs, "count": len(docs)}

@app.post("/query")
async def query_document(query: Query):
    """
    Query all uploaded documents
    Returns answer only
    """
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet."
        )
    
    response = handle_query(query.question)
    return response

@app.delete("/clear")
async def clear_documents():
    """
    Clear all uploaded documents and index
    """
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    
    return {"message": "All documents and index cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)