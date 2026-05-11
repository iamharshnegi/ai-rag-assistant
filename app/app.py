import os 
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, ChatPromptTemplate
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.readers.file import PDFReader, DocxReader, FlatReader

# Load environment variables
load_dotenv()

# LLM config (UPDATED → Mistral)
Settings.llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HF_TOKEN"),
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

PERSIST_DIR = "./db"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# ============= INGESTION PIPELINE =============

def data_ingestion():
    documents = []
    file_count = 0
    
    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        
        if not os.path.isfile(file_path):
            continue
        
        print(f"\n📄 Processing: {file}")
        docs = []
        
        try:
            if file.endswith(".pdf"):
                docs = PDFReader().load_data(file_path)
            elif file.endswith(".docx"):
                docs = DocxReader().load_data(file_path)
            elif file.endswith(".txt"):
                docs = FlatReader().load_data(file_path)
            else:
                print(f"⚠️ Skipping unsupported file: {file}")
                continue
            
            for doc in docs:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["filename"] = file
                if "page_label" not in doc.metadata:
                    doc.metadata["page_label"] = "N/A"
            
            documents.extend(docs)
            file_count += 1
            print(f"✅ Loaded {len(docs)} chunks from {file}")
            
        except Exception as e:
            print(f"❌ Error processing {file}: {str(e)}")
            continue
    
    if not documents:
        print("⚠️ No documents loaded!")
        return
    
    print(f"\n🎉 Total documents loaded: {file_count}")
    print(f"📊 Total chunks: {len(documents)}")
    
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"✅ Index persisted to {PERSIST_DIR}\n")

# ============= RETRIEVAL PIPELINE =============

def expand_query(query):
    expansions = {
        "how is": ["calculation", "formula", "method", "process"],
        "how": ["calculation", "formula", "method", "process", "explained"],
        "interest": ["interest rate", "rate", "calculation", "formula"],
        "total": ["total amount", "sum", "total value"],
        "calculate": ["formula", "calculation", "method", "compute"],
    }
    
    expanded = query.lower()
    for key, values in expansions.items():
        if key in expanded:
            expanded += " " + " ".join(values)
    
    return expanded

def handle_query(query):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    except Exception as e:
        return {
            "answer": "Error loading index. Please upload documents first.",
            "error": str(e)
        }
    
    expanded_query = expand_query(query)
    
    chat_text_qa_msgs = [
        (
            "user",
            """You are a financial and document analysis expert.

TASK: Answer the user's question accurately and thoroughly.

GUIDELINES:
1. Base your answer ONLY on the provided context
2. If the question asks "how is X calculated":
   ✓ Provide the formula
   ✓ Explain each component
   ✓ Show step-by-step calculation if example exists
3. For calculation questions, include:
   - Formula/Method
   - Variables used
   - Values from document
   - Step-by-step working
   - Final result
4. Use numerical examples from the document
5. If information spans multiple sections, synthesize it
6. Do NOT guess or make assumptions
7. If you cannot find the answer, respond: "I cannot find this information in the uploaded documents."

CONTEXT (from documents):
{context_str}

USER QUESTION:
{query_str}

ANSWER:"""
        )
    ]
    
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    query_engine = index.as_query_engine(
        text_qa_template=text_qa_template,
        similarity_top_k=10,
        response_mode="tree_summarize"
    )
    
    response = query_engine.query(expanded_query)
    answer_text = response.response if hasattr(response, "response") else str(response)
    
    return {
        "answer": answer_text
    }