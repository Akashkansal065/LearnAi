import streamlit as st
import tempfile
import asyncio
from pathlib import Path
from typing import List
# Replace with actual filename
from RAG_PDF import upload_markdown, ask_question, modelNames
from fastapi import UploadFile

st.set_page_config(page_title="Doc Uploader + QA", layout="wide")

st.title("📄 Document OCR + Markdown + QA using LangChain + Ollama")

uploaded_files = st.file_uploader("Upload image-based PDFs or images",
                                  accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg"])

if uploaded_files:
    if st.button("Process and Upload to Vector DB"):
        with st.spinner("Processing files..."):
            # Convert Streamlit UploadedFile to FastAPI-style UploadFile
            temp_files: List[UploadFile] = []
            for file in uploaded_files:
                suffix = Path(file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                fastapi_upload = UploadFile(
                    filename=file.name, file=open(tmp_path, "rb"))
                temp_files.append(fastapi_upload)

            result = asyncio.run(upload_markdown(temp_files))
            st.success(result["message"])

st.markdown("---")
st.header("💬 Ask a Question")
user_query = st.text_input(
    "What do you want to know from the uploaded documents?")

if user_query:
    with st.spinner("Searching answers..."):
        response = ask_question(user_query)
        st.markdown(f"**Answer:** {response['answer']}")

        if response['sources']:
            st.markdown("**Sources:**")
            for src in response['sources']:
                st.markdown(f"- {src}")
