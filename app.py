import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide", 
    page_title="Document Chat",
    page_icon="📄"
)

import os
import tempfile
import time
import base64
from rag import process_uploaded_pdf as puf, handle_query as hq

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

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf">'

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.uploaded_pdf = tmp_file.name

        try:
            with st.spinner("Processing PDF..."):
                process_result = puf(st.session_state.uploaded_pdf)
                st.session_state.processed = True
                st.success("PDF processed successfully!")
        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")

# Main content area
col1, col2 = st.columns([0.5, 0.5], gap="medium")

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
            col1, col2 = st.columns([0.3, 0.7])
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
                result = hq(query)
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
            with st.container(border=True):
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.markdown(f"### {note['name']}")
                    st.markdown(f"**Question:** {note['query']}")
                    st.markdown(f"**Response:** {note['response']}")
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{idx}"):
                        del st.session_state.notes[idx]
                        st.rerun()
    else:
        st.info("Saving Notes Feature is not available cause of streamlit constraints, I have built a Flask App but not deployed cause of resource limitation (512 MB of RAM in free tier), ⚠️CAN GIVE A LIVE DEMO⚠️")

# Health check
if st.button("Check System Health"):
    st.json({"status": "healthy", "timestamp": time.time()})