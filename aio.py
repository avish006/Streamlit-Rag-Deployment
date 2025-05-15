import streamlit as st
st.set_page_config(layout="centered", page_title="Document Chat", page_icon="📄")

import os
import tempfile
import time
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_state():
    defaults = {
        'processed': False,
        'chat_history': [],
        'uploaded_pdf': None,
        'notes': [],
        'show_save_dialog': False,
        'current_note': None,
        'processed_data': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

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

@st.cache_data
def save_uploaded_file(uploaded_file):
    temp_dir = "uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

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
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separator='\n')
    return splitter.split_text(text)

def store_embeddings_faiss(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

def hybrid_search(query, index, chunks, alpha=0.5, top_k=5):
    if not chunks or index is None:
        return []
    
    query_embedding = models['sentence_model'].encode(query, normalize_embeddings=True)
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    
    tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(word_tokenize(query.lower()))
    
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
        prompt = f"""Query: {user_query}\nContext: {retrieved}\nYou are an expert assistant. Provide a clear, concise response in Markdown."""
        
        response = models['openai_client'].chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1600,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return f"Error: {str(e)}"

def handle_query(user_query):
    if not st.session_state.processed_data:
        return {"error": "No PDF processed"}
    _, chunks, _, index = st.session_state.processed_data
    response = query_rag(user_query, index, chunks)
    return {"response": response}

st.title("📄 PDF Chat Assistant")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], help="Upload a PDF document to start chatting")

if uploaded_file and not st.session_state.processed:
    file_path = save_uploaded_file(uploaded_file)
    st.session_state.uploaded_pdf = file_path
    with st.spinner("Processing PDF..."):
        st.session_state.processed_data = load_or_process_pdf(file_path)
        st.session_state.processed = True
        st.session_state.chat_history = []
        st.success("PDF processed successfully!")

chat_container = st.container(height=600)

for idx, entry in enumerate(st.session_state.chat_history):
    with chat_container:
        with st.chat_message("user"):
            st.markdown(entry['query'])
        with st.chat_message("assistant"):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(entry['response'])
            with col_b:
                if st.button("🗕️ Save", key=f"save_{idx}", use_container_width=True):
                    st.session_state.current_note = entry
                    st.session_state.show_save_dialog = True

if st.session_state.show_save_dialog:
    with st.form("Save Note"):
        note_name = st.text_input("Note name", value=st.session_state.current_note['query'][:50])
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            if st.form_submit_button("_ Save"):
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
        result = handle_query(query)
        st.session_state.chat_history.append({
            'query': query,
            'response': result.get('response', 'Error or empty response')
        })
        with assistant_placeholder.container():
            with st.chat_message("assistant"):
                st.markdown(result.get('response', 'Error or empty response'))

if st.session_state.notes:
    for idx, note in enumerate(st.session_state.notes):
        with st.expander(note['name'], expanded=False):
            st.markdown(f"**Question:** {note['query']}")
            st.markdown(f"**Response:** {note['response']}")
            if st.button("🗑️ Delete", key=f"delete_{idx}"):
                del st.session_state.notes[idx]
                st.experimental_rerun()
else:
    st.info("Viewing pdfs in real time are not featured in this streamlit demo cause of streamlit usage constraints, 🟢Originally the code was written in flask (not streamlit) but cause of deployment resource constraints used streamlit,  ⚠️To Get Live Demo Contact Me!⚠️   ")


if st.button("Check System Health"):
    st.json({"status": "healthy", "timestamp": time.time()})
