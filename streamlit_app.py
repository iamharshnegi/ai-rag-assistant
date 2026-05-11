import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

def streamlit_ui():
    st.set_page_config(page_title="Multi-Doc Chat", layout="wide")
    
    st.title("🤖 Chat with Multiple Documents")
    st.markdown("Upload one or more PDFs, DOCX, or TXT files and ask questions about them!")
    
    icons = {"assistant": "🤖", "user": "👤"}
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hello! Upload one or more documents and ask me anything about them."
            }
        ]
    
    # ============= SIDEBAR: FILE UPLOAD & MANAGEMENT =============
    with st.sidebar:
        st.title("📁 Document Manager")
        
        # Show uploaded documents
        try:
            docs_response = requests.get(f"{BASE_URL}/documents", timeout=10)
            if docs_response.status_code == 200:
                docs_data = docs_response.json()
                docs_list = docs_data.get("documents", [])
                doc_count = docs_data.get("count", 0)
                
                if doc_count > 0:
                    st.success(f"✅ {doc_count} document(s) uploaded")
                    with st.expander("📋 View Uploaded Documents"):
                        for doc in docs_list:
                            st.write(f"📄 {doc}")
                else:
                    st.info("No documents uploaded yet")
        except:
            pass
        
        st.divider()
        
        # Upload new files
        st.subheader("📤 Upload New Documents")
        uploaded_files = st.file_uploader(
            "Select one or multiple files (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("📤 Submit & Process", use_container_width=True) and uploaded_files:
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                try:
                    files = [
                        (
                            "files",
                            (file.name, file.getvalue(), file.type)
                        )
                        for file in uploaded_files
                    ]
                    
                    response = requests.post(
                        f"{BASE_URL}/upload",
                        files=files,
                        timeout=180
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(result["message"])
                        st.json({
                            "files_uploaded": result.get("files", [])
                        })
                    else:
                        st.error(f"Upload failed: {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend not running. Start FastAPI:\n`python main.py`")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # Clear all documents
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            try:
                response = requests.delete(f"{BASE_URL}/clear", timeout=10)
                if response.status_code == 200:
                    st.success("All documents cleared!")
            except:
                st.error("Error clearing documents")
    
    # ============= MAIN CHAT AREA =============
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons.get(message["role"], "💬")):
            st.write(message["content"])
    
    # ============= USER INPUT =============
    user_prompt = st.chat_input("Ask me anything about your documents...")
    
    if user_prompt:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.write(user_prompt)
        
        # Query backend
        with st.spinner("🔍 Searching documents..."):
            try:
                response = requests.post(
                    f"{BASE_URL}/query",
                    json={"question": user_prompt},
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer generated")
                    
                    # Display assistant response
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(answer)
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                
                else:
                    st.error(f"Query failed: {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend not running. Start FastAPI:\n`python main.py`")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    streamlit_ui()