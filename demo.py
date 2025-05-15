import streamlit as st
st.set_page_config(
    page_title="PDF Viewer Demo",
    layout="centered",
    page_icon="📄"
)

import tempfile
import base64

import streamlit.components.v1 as components

# Set page config

st.title("📄 PDF Upload & Viewer Demo")

# Function to render PDF safely using base64 + iframe (works on deployed apps)
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="700"
            type="application/pdf"
            style="border: none;"
        ></iframe>
    """
    components.html(pdf_display, height=700)

# Upload area
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    st.success("PDF uploaded successfully!")
    display_pdf(tmp_path)
else:
    st.info("Please upload a PDF file to view it here.")
