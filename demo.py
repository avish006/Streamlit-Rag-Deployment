import streamlit as st
import base64

def display_pdf(file_bytes):
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    return f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf">'

st.set_page_config(layout="wide", page_title="PDF Viewer")

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file:
    pdf_bytes = uploaded_file.getvalue()
    st.session_state.uploaded_pdf = pdf_bytes

if 'uploaded_pdf' in st.session_state:
    st.markdown(display_pdf(st.session_state.uploaded_pdf), unsafe_allow_html=True)
else:
    st.info("Upload a PDF to view it here.")
