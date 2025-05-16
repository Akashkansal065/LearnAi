# fastapi_langchain_pipeline.py

import pathlib
import shutil
from ollama_ocr import OCRProcessor
import uvicorn
import hashlib
import uuid
import os
from fastapi import FastAPI, UploadFile, File
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

from ollamaApi import OllamaClient

app = FastAPI()
ollama = OllamaClient()
UPLOAD_DIR = "./uploaded_md"
OUTPUT_DIR = "./output_md"
VECTOR_STORE_DIR = "./chroma_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_duplicate(doc_chunk, db, threshold=0.9):
    results = db.similarity_search(doc_chunk.page_content, k=1)
    if results:
        similarity_score = results[0].metadata.get("similarity", 1.0)
        return similarity_score >= threshold
    return False


def load_and_split_markdown(directory: str):
    loader = DirectoryLoader(
        path=directory,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def create_vector_store(docs, persist_directory: str):
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2")
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings)
    doc_ids = [str(uuid.uuid4()) for _ in range(len(docs))]
    db.add_documents(documents=docs, ids=doc_ids)
    # new_docs = []
    # new_ids = []
    # for doc in docs:
    #     first_chunk = RecursiveCharacterTextSplitter(
    #         chunk_size=500, chunk_overlap=0).split_documents([doc])[0]
    #     hash_input = first_chunk.page_content.strip().encode("utf-8")
    #     doc_id = str(uuid.UUID(hashlib.md5(hash_input).hexdigest()))

    #     if not is_duplicate(first_chunk, db):
    #         new_docs.append(doc)
    #         new_ids.append(doc_id)

    # if new_docs:
    #     db.add_documents(documents=new_docs, ids=new_ids)

    return db


def create_vector_store_reupload(docs, persist_directory: str):
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2")
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings)
    doc_ids = [str(uuid.uuid4()) for _ in range(len(docs))]
    db.add_documents(documents=docs, ids=doc_ids)
    new_docs = []
    new_ids = []
    for doc in docs:
        first_chunk = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0).split_documents([doc])[0]
        hash_input = first_chunk.page_content.strip().encode("utf-8")
        doc_id = str(uuid.UUID(hashlib.md5(hash_input).hexdigest()))

        if not is_duplicate(first_chunk, db):
            new_docs.append(doc)
            new_ids.append(doc_id)

    if new_docs:
        db.add_documents(documents=new_docs, ids=new_ids)

    return db


def create_qa_chain(vectorstore):
    llm = OllamaLLM(model="qwen3:14b")
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


@app.get('/modellist')
async def modelNames():
    model_name = ollama.list_models()
    return model_name


@app.post("/upload/")
async def upload_markdown(files: List[UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    processor = OCRProcessor(model_name='llama3.2-vision', max_workers=1)
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        # Save uploaded file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = processor.process_image(
            image_path=file_path, format_type="markdown")
        output_path = pathlib.Path(OUTPUT_DIR) / f"{file.filename}_output.md"
        output_path.write_text(result, encoding="utf-8")
        os.remove(file_path)
    docs = load_and_split_markdown(OUTPUT_DIR)
    create_vector_store(docs, persist_directory=VECTOR_STORE_DIR)
    try:
        # os.remove(file_path)
        os.remove(output_path)
    except FileNotFoundError:
        pass
    return {"message": f"Uploaded and processed {len(files)} file(s)."}


@app.get("/ask")
def ask_question(query: str):
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2")
    db = Chroma(persist_directory=VECTOR_STORE_DIR,
                embedding_function=embeddings)
    qa_chain = create_qa_chain(db)
    result = qa_chain.invoke(query)

    sources = [doc.metadata.get("source")
               for doc in result["source_documents"]]
    return {
        "answer": result["result"],
        "sources": sources
    }


if __name__ == "__main__":
    uvicorn.run("Load_Store_Retrieve:app",
                host="0.0.0.0", port=8070, reload=True)
  # , workers=4)
