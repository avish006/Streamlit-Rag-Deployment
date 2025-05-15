import streamlit as st
import os
import shutil
import uuid

st.set_page_config(page_title="PDF Viewer", layout="centered")

st.title("📄 Secure PDF Viewer with PDF.js")

# Folder where temporary uploaded PDFs will be stored
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Save the uploaded file to a safe location
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    # Unique filename to avoid collisions
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Construct iframe URL using pdf.js from a CDN
    viewer_url = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={st.secrets['pdf_cdn_host']}/{file_id}.pdf"

    st.markdown("### 📑 Document Preview:")
    st.components.v1.iframe(viewer_url, height=800, width=700)
else:
    st.info("Upload a PDF file to view it here.")
