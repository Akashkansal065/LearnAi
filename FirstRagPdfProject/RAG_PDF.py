import fitz
import re
import pathlib
import shutil
import os
import uuid
import hashlib
from typing import List
from ollama_ocr import OCRProcessor
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from ollamaApi import OllamaClient

ollama = OllamaClient()
UPLOAD_DIR = "./uploaded_md"
OUTPUT_DIR = "./output_md"
VECTOR_STORE_DIR = "./chroma_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DocumentProcessor:
    def __init__(self, vector_store_dir: str, upload_dir: str, output_dir: str,
                 model_name: str = 'llama3.2-vision', embedding_model: str = 'snowflake-arctic-embed2'):
        self.vector_store_dir = vector_store_dir
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.processor = OCRProcessor(
            model_name=model_name, max_workers=1)
        self.embedding_model = embedding_model

    def get_pdf_title(self, file_path):
        try:
            # Open the PDF file
            doc = fitz.open(file_path)

            # Extract the metadata of the PDF
            metadata = doc.metadata

            # Get the title from the metadata
            title = metadata.get("title", "No title found")

            # Optionally, print other metadata if needed
            print(f"Title: {title}")
            return title

        except Exception as e:
            print(f"Error processing the PDF: {e}")
            return None

    def is_semantic_duplicate(self, doc, db, threshold=0.9):
        results = db.similarity_search(doc.page_content, k=1)
        if results:
            return results[0].page_content.strip() == doc.page_content.strip()
        return False

    def load_and_split_markdown(self, directory: str):
        loader = DirectoryLoader(
            path=directory,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(documents)

    def create_vector_store(self, docs, persist_directory: str):
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        db = Chroma(persist_directory=persist_directory,
                    embedding_function=embeddings)
        doc_ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        db.add_documents(documents=docs, ids=doc_ids)
        return db

    def create_vector_store_dedup(self, docs, persist_directory: str):
        embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2")
        db = Chroma(persist_directory=persist_directory,
                    embedding_function=embeddings)

        new_docs = []
        for doc in docs:
            if not self.is_semantic_duplicate(doc, db):
                new_docs.append(doc)

        if new_docs:
            doc_ids = [str(uuid.uuid4()) for _ in range(len(new_docs))]
            db.add_documents(documents=new_docs, ids=doc_ids)

        return db

    def clean_response(self, text):
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned_text.strip()

    def create_qa_chain(self, vectorstore):
        llm = OllamaLLM(model="qwen3:14b")
        retriever = vectorstore.as_retriever()
        return RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    def upload_markdown(self, files):
        os.makedirs(self.upload_dir, exist_ok=True)
        for file in files:
            # file_path = os.path.join(self.upload_dir, file.filename)
            # filename = os.path.basename(file)
            # file_path = os.path.join(self.upload_dir, filename)
            # with open(file_path, "wb") as buffer:
            #     shutil.copyfileobj(file.file, buffer)
            result = self.processor.process_image(
                image_path=file, format_type="markdown")
            output_path = pathlib.Path(
                self.output_dir) / f"{file}_output.md"
            output_path.write_text(result, encoding="utf-8")
            # os.remove(file)

        docs = self.load_and_split_markdown(self.output_dir)
        self.create_vector_store_dedup(
            docs, persist_directory=self.vector_store_dir)
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass
        return {"message": f"Uploaded and processed {len(files)} file(s)."}

    def ask_question(self, query: str):
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        db = Chroma(persist_directory=self.vector_store_dir,
                    embedding_function=embeddings)
        qa_chain = self.create_qa_chain(db)
        result = qa_chain.invoke(query)

        sources = [doc.metadata.get("source")
                   for doc in result["source_documents"]]
        return {
            "answer": self.clean_response(result['answer']),
            "sources": sources
        }


document_processor = DocumentProcessor(
    vector_store_dir=VECTOR_STORE_DIR,
    upload_dir=UPLOAD_DIR,
    output_dir=OUTPUT_DIR,
    model_name='qwen3'
)

# For uploading markdown files
# response = document_processor.upload_markdown(path)

# For asking a question
response = document_processor.ask_question(
    "Can you create a javascript Function to check current date?")
print(response)
