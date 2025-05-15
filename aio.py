import streamlit as st
st.set_page_config(
    layout="wide", 
    page_title="Document Chat",
    page_icon="📄"
)
import os
import tempfile
import time
import base64
import re
import httpx
import pytesseract
import numpy as np
import hashlib
import pickle
import logging
from PIL import Image
from io import BytesIO
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import faiss
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import nltk
import easyocr

try:
    import pymupdf as fitz
except ImportError:
    import fitz

# MUST BE FIRST STREAMLIT COMMAND


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_pdf' not in st.session_state:
    st.session_state.uploaded_pdf = None
if 'notes' not in st.session_state:
    st.session_state.notes = []
if 'show_save_dialog' not in st.session_state:
    st.session_state.show_save_dialog = False
if 'current_note' not in st.session_state:
    st.session_state.current_note = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Initialize models (cached)
@st.cache_resource
def load_models():
    nltk.download('punkt')
    return {
        'sentence_model': SentenceTransformer('BAAI/bge-small-en-v1.5'),
        'clip_model': CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
        'clip_processor': CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        'ocr_reader': easyocr.Reader(['en'], gpu=False),
        'openai_client': OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY"), http_client=httpx.Client(timeout=30.0))
    }

models = load_models()
# PDF Processing Functions
def process_page(_args):
    page_num, pdf_path = _args
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    content = page.get_text()
    doc.close()
    return content

def extract_content_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join([process_page((i, pdf_path)) for i in range(len(doc))])

def recursive_chunking(text, chunk_size=500, overlap=100):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separator='\n'
    )
    return text_splitter.split_text(text)

def store_embeddings_faiss(embeddings):
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

def hybrid_search(query, index, chunks, alpha=0.5, top_k=5):
    if not chunks or index is None:
        return []
    
    # Vector search
    query_embedding = models['sentence_model'].encode(query, normalize_embeddings=True)
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    
    # BM25 search
    tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(word_tokenize(query.lower()))
    
    # Combine results
    combined = []
    for i in range(top_k):
        if I[0][i] < len(bm25_scores):
            score = alpha * bm25_scores[I[0][i]] + (1 - alpha) * (1 / (1 + D[0][i]))
            combined.append((I[0][i], score))
    
    combined.sort(key=lambda x: x[1], reverse=True)
    return [chunks[idx] for idx, _ in combined[:top_k]]

def get_pdf_hash(pdf_path):
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_or_process_pdf(pdf_path):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    pdf_hash = get_pdf_hash(pdf_path)
    cache_file = f"{cache_dir}/{pdf_hash}.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    text = extract_content_from_pdf(pdf_path)
    chunks = recursive_chunking(text)
    embeddings = models['sentence_model'].encode(chunks, normalize_embeddings=True)
    index = store_embeddings_faiss(embeddings)

    with open(cache_file, "wb") as f:
        pickle.dump((text, chunks, embeddings, index), f)

    return text, chunks, embeddings, index

def query_rag(user_query, index, chunks):
    try:
        retrieved = hybrid_search(user_query, index, chunks)
        
        prompt = f"""Query: {user_query}
        Context: {retrieved}
        You are an expert assistant. Provide a clear, concise response.
        Respond in clean Markdown format with proper formatting.
        """
        
        response = models['openai_client'].chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1600,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return f"Error processing query: {str(e)}"

def process_uploaded_pdf(pdf_path):
    try:
        processed_data = load_or_process_pdf(pdf_path)
        st.session_state.processed_data = processed_data
        st.session_state.chat_history = []
        return {"status": "success"}
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        return {"error": str(e)}

def handle_query(user_query):
    if 'processed_data' not in st.session_state or not st.session_state.processed_data:
        return {"error": "No PDF processed"}
    
    _, chunks, _, index = st.session_state.processed_data
    response = query_rag(user_query, index, chunks)
    return {"response": response}

# UI Components
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.uploaded_pdf = tmp_file.name

        try:
            with st.spinner("Processing PDF..."):
                process_result = process_uploaded_pdf(st.session_state.uploaded_pdf)
                st.session_state.processed = True
                st.success("PDF processed successfully!")
        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")

# Main content area
col1, col2 = st.columns([0.0, 1], gap="small")

# Left pane - PDF Viewer
with col1:
    st.header("Document Viewer")
    if st.session_state.uploaded_pdf:
        st.markdown(display_pdf(st.session_state.uploaded_pdf), unsafe_allow_html=True)
    else:
        st.info("Upload a PDF document to get started")

# Right pane - Chat Interface
with col2:
    st.header("Chat Interface")
    chat_container = st.container(height=600)

    for idx, entry in enumerate(st.session_state.chat_history):
        with chat_container:
            with st.chat_message("user"):
                st.markdown(entry['query'])

            if entry['response']:
                with st.chat_message("assistant"):
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.markdown(entry['response'])
                    with col_b:
                        if st.button("🗕️ Save", key=f"save_{idx}", use_container_width=True):
                            st.session_state.current_note = {
                                'query': entry['query'],
                                'response': entry['response']
                            }
                            st.session_state.show_save_dialog = True

    if st.session_state.show_save_dialog:
        with st.form("Save Note"):
            note_name = st.text_input("Note name", value=st.session_state.current_note['query'][:50])
            col1, col2 = st.columns([0.0, 1.0])
            with col1:
                if st.form_submit_button("🗕️ Save"):
                    st.session_state.notes.append({
                        'name': note_name,
                        'query': st.session_state.current_note['query'],
                        'response': st.session_state.current_note['response']
                    })
                    st.session_state.show_save_dialog = False
                    st.rerun()
            with col2:
                if st.form_submit_button("❌ Cancel"):
                    st.session_state.show_save_dialog = False

    query = st.chat_input("Ask about the document...")
    if query and st.session_state.processed:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
            assistant_placeholder = st.empty()

        with st.spinner("Analyzing document..."):
            try:
                result = handle_query(query)
                st.session_state.chat_history.append({
                    'query': query,
                    'response': result.get('response', 'Error or empty response')
                })
                with assistant_placeholder.container():
                    with st.chat_message("assistant"):
                        st.markdown(result.get('response', 'Error or empty response'))
            except Exception as e:
                st.error(f"Query failed: {str(e)}")

# Notes tab
with st.expander("📚 Saved Notes", expanded=False):
    if st.session_state.notes:
        for idx, note in enumerate(st.session_state.notes):
            # Track which note is in edit mode using session_state
            edit_key = f"edit_mode_{idx}"
            if edit_key not in st.session_state:
                st.session_state[edit_key] = False
            
            with st.expander(note['name'], expanded=False):
                if st.session_state[edit_key]:
                    # Editable form fields
                    new_name = st.text_input("Note name", value=note['name'], key=f"name_{idx}")
                    new_query = st.text_area("Question", value=note['query'], key=f"query_{idx}")
                    new_response = st.text_area("Response", value=note['response'], key=f"response_{idx}")

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("💾 Save Changes", key=f"save_{idx}"):
                            # Save edits back to session_state.notes
                            st.session_state.notes[idx] = {
                                'name': new_name.strip() or note['name'],  # fallback to old name if empty
                                'query': new_query.strip() or note['query'],
                                'response': new_response.strip() or note['response']
                            }
                            st.session_state[edit_key] = False
                            st.experimental_rerun()
                    with col2:
                        if st.button("❌ Cancel", key=f"cancel_{idx}"):
                            st.session_state[edit_key] = False
                            st.experimental_rerun()
                else:
                    # Display saved note
                    st.markdown(f"**Question:** {note['query']}")
                    st.markdown(f"**Response:** {note['response']}")
                    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
                    with col1:
                        if st.button("✏️ Edit", key=f"edit_{idx}"):
                            st.session_state[edit_key] = True
                            st.experimental_rerun()
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_{idx}"):
                            del st.session_state.notes[idx]
                            st.experimental_rerun()
    else:
        st.info("Saving Notes Feature is not available due to Streamlit constraints, Contact me for a Full Live Demo")


# Health check
if st.button("Check System Health"):
    st.json({"status": "healthy", "timestamp": time.time()})
