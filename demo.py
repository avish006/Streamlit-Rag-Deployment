import streamlit as st
import base64

st.set_page_config(page_title="PDF Viewer", layout="wide")

def display_pdf(pdf_bytes):
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'''
        <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" type="application/pdf"></iframe>
    '''
    return pdf_display

st.title("📄 PDF Viewer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    st.markdown(display_pdf(pdf_bytes), unsafe_allow_html=True)
else:
    st.info("Please upload a PDF to view it here.")
