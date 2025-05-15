# rag.py (Streamlit Version)
import os
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
import streamlit as st
import easyocr
try:
    import pymupdf as fitz
except ImportError:
    import fitz

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models (cached)
@st.cache_resource
def load_models():
    nltk.download('punkt')
    return {
        'sentence_model': SentenceTransformer('BAAI/bge-small-en-v1.5'),
        'clip_model': CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
        'clip_processor': CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        'ocr_reader': easyocr.Reader(['en'], gpu=False),
        'openai_client': OpenAI(base_url="https://openrouter.ai/api/v1",api_key="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",http_client=httpx.Client(timeout=30.0))
    }

models = load_models()

def process_page(_args):
    """Simplified page processing for Streamlit"""
    page_num, pdf_path = _args
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    content = page.get_text()
    doc.close()
    return content

def extract_content_from_pdf(pdf_path):
    """Streamlit-optimized PDF extraction"""
    with fitz.open(pdf_path) as doc:
        return "\n".join([process_page((i, pdf_path)) for i in range(len(doc))])

def recursive_chunking(text, chunk_size=500, overlap=100):
    """Text chunking with progress"""
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separator='\n'
    )
    return text_splitter.split_text(text)

def store_embeddings_faiss(embeddings):
    """Create FAISS index"""
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

def hybrid_search(query, index, chunks, alpha=0.5, top_k=5):
    """Simplified hybrid search for Streamlit"""
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
    """Generate PDF hash for caching"""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_or_process_pdf(pdf_path):
    """Streamlit-optimized PDF processing with caching"""
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
    """Simplified RAG query handler for Streamlit"""
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
    """Streamlit PDF processing entry point"""
    try:
        processed_data = load_or_process_pdf(pdf_path)
        st.session_state.processed_data = processed_data
        st.session_state.chat_history = []
        return {"status": "success"}
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        return {"error": str(e)}

def handle_query(user_query):
    """Streamlit query handling entry point"""
    if 'processed_data' not in st.session_state or not st.session_state.processed_data:
        return {"error": "No PDF processed"}
    
    _, chunks, _, index = st.session_state.processed_data
    response = query_rag(user_query, index, chunks)
    return {"response": response}
