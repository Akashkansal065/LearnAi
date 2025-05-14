# chat_logic.py

import re
import os
import time
import traceback
import asyncio
from typing import Dict, List
from dotenv import load_dotenv
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import ollama  # For listing models

# Langchain Core/Community imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from ollama_ocr import OCRProcessor
# Ollama/Chroma imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient

load_dotenv()  # Load environment variables from .env file
try:

    OCR_PROCESSOR_AVAILABLE = True
    print("Successfully imported OCRProcessor from ollama_ocr.py")
except ImportError:
    print("Warning: Could not import OCRProcessor from ollama_ocr.py. Custom OCR mode will use a dummy.")
    OCR_PROCESSOR_AVAILABLE = False

    class OCRProcessor:  # Dummy class if import fails
        def __init__(self, model_name): self.model_name = model_name; print(
            f"Dummy OCRProcessor initialized with {model_name}")
        # Modify dummy process_image to accept image_path
        def process_image(self, image_path): print(
            f"Dummy OCRProcessor called for {image_path}. Returning no text."); return ""


# --- Default Model Constants --- (Remain unchanged)
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = 'llama3:8b'
DEFAULT_OCR_MODEL_NAME = 'llava:latest'
CHROMA_STORE_PATH = './chroma_store'
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11")

# Create Chroma directory if it doesn't exist
if not os.path.exists(CHROMA_STORE_PATH):
    try:
        os.makedirs(CHROMA_STORE_PATH)
        print(f"Created Chroma directory: {CHROMA_STORE_PATH}")
    except OSError as e:
        print(f"Error creating Chroma directory {CHROMA_STORE_PATH}: {e}")


def sanitize_collection_name(name: str) -> str:
    # Replace invalid characters (e.g., space) with underscore
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.strip())
    # Ensure it's within allowed length
    name = name[:63].strip('_')
    # Ensure starts/ends with alphanumeric character
    if not name[0].isalnum():
        name = 'c' + name
    if not name[-1].isalnum():
        name = name + 'c'
    return name

# --- Helper functions (can be async if they do I/O) ---


# async def get_ollama_models_list_async():
#     try:
#         models_data = await asyncio.to_thread(ollama.list)
#         print(f"Fetched {len(models_data.models)} models from Ollama.")
#         return [model.model for model in models_data.models]
#     except Exception as e:
#         print(f"Error fetching Ollama models: {e}. Returning empty list.")
#         return []


async def get_ollama_models_list_async():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            models_data = response.json()
            print(
                f"Fetched {len(models_data['models'])} models from remote Ollama.")
            return [model["name"] for model in models_data["models"]]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}. Returning empty list.")
        return []


async def list_chroma_collections_async():
    try:
        client = await asyncio.to_thread(PersistentClient, path=CHROMA_STORE_PATH)
        collections = await asyncio.to_thread(client.list_collections)
        return collections
    except Exception as e:
        print(
            f"Error listing Chroma collections from {CHROMA_STORE_PATH}: {e}")
        traceback.print_exc()
        return []


# def list_openai_llm_models_static():
#     return ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4o"]


# def list_openai_embedding_models_static():
#     return ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]


# def get_llm_instance(provider: str, model_name: str, temperature: float = 0.1):
#     """Instantiates and returns an LLM based on the provider."""
#     print(
#         f"Getting LLM instance for provider: {provider}, model: {model_name}")
#     if provider == "openai":
#         if not os.getenv("OPENAI_API_KEY"):
#             raise ValueError(
#                 "OPENAI_API_KEY environment variable not set for OpenAI provider.")
#         return ChatOpenAI(model_name=model_name, temperature=temperature)
#     elif provider == "ollama":
#         return ChatOllama(model=model_name, temperature=temperature, base_url=OLLAMA_BASE_URL)
#     else:
#         raise ValueError(f"Unsupported LLM provider: {provider}")


# def get_embedding_instance(provider: str, model_name: str):
#     """Instantiates and returns an embedding model based on the provider."""
#     print(
#         f"Getting Embedding instance for provider: {provider}, model: {model_name}")
#     if provider == "openai":
#         if not os.getenv("OPENAI_API_KEY"):
#             raise ValueError(
#                 "OPENAI_API_KEY environment variable not set for OpenAI provider.")
#         return OpenAIEmbeddings(model=model_name)
#     elif provider == "ollama":
#         return OllamaEmbeddings(model=model_name, base_url=OLLAMA_BASE_URL)
#     else:
#         raise ValueError(f"Unsupported Embedding provider: {provider}")


async def index_document(
    file_path: str,
    file_name: str,
    collection_name: str,
    processing_mode: str = "text",
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ocr_model_name: str = DEFAULT_OCR_MODEL_NAME
) -> tuple[bool, str]:
    """
    Loads, splits, embeds, and stores ONE document's content into a ChromaDB collection.
    Handles PDF, TXT, and Image files. Uses OCRProcessor for images and OCR mode PDFs.
    Returns (success: bool, message: str).
    """
    collection_name = sanitize_collection_name(collection_name)
    print(f"Sanitized collection name: {collection_name}")
    if not collection_name:
        collection_name = "default_collection"
    print(f"\n--- Processing document: {os.path.basename(file_path)} ---")
    print(
        f"Target Collection: {collection_name}, Processing Mode Hint: {processing_mode}")
    documents = []
    file_extension = os.path.splitext(file_path)[1].lower()
    is_image = file_extension in [
        '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']  # Add more if needed

    try:
        # 1. Load Document
        if is_image:
            print(
                f"File type: Image. Using OCRProcessor (Model: {ocr_model_name})")
            if not OCR_PROCESSOR_AVAILABLE:
                return False, "OCRProcessor not available. Cannot process image."
            try:
                processor = OCRProcessor(model_name=ocr_model_name)
                extracted_text = await asyncio.to_thread(processor.process_image, image_path=file_path)
                print(
                    f"OCRProcessor completed. Text length: {len(extracted_text)}")

                if extracted_text:
                    documents = [Document(page_content=extracted_text, metadata={
                        "source": os.path.basename(file_path), "processing_mode": "image_ocr"})]
                else:
                    return False, "OCRProcessor returned empty text for image."
            except Exception as ocr_error:
                traceback.print_exc()
                return False, f"Error during image OCR processing: {ocr_error}"

        elif file_extension == ".pdf":
            if processing_mode == "ocr":
                print(
                    f"File type: PDF (OCR Mode). Using OCRProcessor (Model: {ocr_model_name})")
                if not OCR_PROCESSOR_AVAILABLE:
                    return False, "OCRProcessor not available for PDF OCR."
                try:
                    processor = OCRProcessor(model_name=ocr_model_name)
                    extracted_text = await asyncio.to_thread(processor.process_image, image_path=file_path)
                    print(
                        f"OCRProcessor completed. Text length: {len(extracted_text)}")

                    if extracted_text:
                        documents = [Document(page_content=extracted_text, metadata={
                            "source": os.path.basename(file_path), "processing_mode": "pdf_ocr"})]
                    else:
                        return False, "OCRProcessor returned empty text for PDF."
                except Exception as ocr_error:
                    traceback.print_exc()
                    return False, f"Error during PDF OCR processing: {ocr_error}"
            else:
                print("File type: PDF (Text Mode). Using PyPDFLoader.")
                loader = PyPDFLoader(file_path)
                loaded_docs = await asyncio.to_thread(loader.load)
                documents = loaded_docs

        elif file_extension == ".txt":
            print("File type: TXT. Using TextLoader.")
            loader = TextLoader(file_path, encoding='utf-8')
            loaded_docs = await asyncio.to_thread(loader.load)
            documents = loaded_docs

        else:
            return False, f"Unsupported file type '{file_extension}'."

        # --- Common Steps (Splitting, Embedding, Storing) ---
        if not documents:
            return False, "No content loaded or extracted from the document."

        print(f"Loaded {len(documents)} Langchain Document object(s).")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        chunks = await asyncio.to_thread(text_splitter.split_documents, documents)

        if not chunks:
            return False, "Document was loaded but could not be split into chunks."

        print(f"Split into {len(chunks)} chunks.")

        print(f"Initializing embeddings with model: {embedding_model_name}")
        embeddings = OllamaEmbeddings(
            model=embedding_model_name, base_url=OLLAMA_BASE_URL)

        print(
            f"Storing chunks in Chroma collection '{collection_name}' at {CHROMA_STORE_PATH}")
        await asyncio.to_thread(
            Chroma.from_documents,
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=CHROMA_STORE_PATH
        )

        success_msg = f"Successfully added chunks from '{os.path.basename(file_path)}' to collection '{collection_name}'."
        print(success_msg)
        return True, success_msg

    except Exception as e:
        traceback.print_exc()
        return False, f"Unhandled error while processing document: {e}"


async def get_rag_answer_async(
    question: str,
    collection_name: str = "default_collection",
    llm_model_name: str = DEFAULT_LLM_MODEL,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    chroma_path: str = "./chroma_store"
) -> str:
    print(f"\n--- Generating RAG answer (Async) ---")
    print(f"Question: '{question}', Collection: '{collection_name}'")
    if not collection_name:
        collection_name = "default_collection"

    try:
        # Assume sync components run in threadpool if needed
        # print(OLLAMA_BASE_URL)
        embeddings = OllamaEmbeddings(
            model=embedding_model_name, base_url=OLLAMA_BASE_URL)
        vectorstore = await asyncio.to_thread(
            Chroma,
            persist_directory=chroma_path,
            embedding_function=embeddings,
            collection_name=collection_name
        )

        # Sync attribute access
        count = await asyncio.to_thread(vectorstore._collection.count)
        if count == 0:
            return f"Error: Collection '{collection_name}' is empty."
        print(
            f"Loaded collection '{collection_name}' with {count} embedded chunks.")

        retriever = vectorstore.as_retriever(
            search_kwargs={'k': 3})  # This might be sync setup

        template = """You are an assistant for question-answering tasks based ONLY on the following context. If you don't know the answer from the context, say that you don't know. Do not use outside knowledge. Keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        # Assume ChatOllama supports async via .ainvoke
        llm = ChatOllama(model=llm_model_name, temperature=0.1,
                         base_url=OLLAMA_BASE_URL)
        rag_chain = ({"context": retriever, "question": RunnablePassthrough(
        )} | prompt | llm | StrOutputParser())

        print("Invoking RAG chain (async)...")
        answer = await rag_chain.ainvoke(question)  # Use ainvoke
        print(f"RAG Answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error in RAG (Async): {e}")
        traceback.print_exc()
        return f"Error: RAG failed. {e}"


async def get_general_answer_async(
    question: str,
    chat_history_list: list,
    llm_model_name: str = DEFAULT_LLM_MODEL,
) -> str:
    """ (Async Non-RAG Answer) Generates an answer using only the LLM and chat history. """
    print(f"\n--- Generating General answer (Async) ---")
    print(f"Question: '{question}'")
    try:
        # Assume async support
        llm = ChatOllama(model=llm_model_name, temperature=0.1,
                         base_url=OLLAMA_BASE_URL)
        general_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Answer the user's
             question based on the conversation history if relevant."""),
            MessagesPlaceholder(variable_name="chat_history_for_prompt"),
            ("human", "{current_question_for_prompt}")
        ])
        general_chain = (general_prompt | llm | StrOutputParser())

        formatted_history_for_prompt = []
        for msg in chat_history_list:
            if msg["role"] == "user":
                formatted_history_for_prompt.append(
                    HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history_for_prompt.append(
                    AIMessage(content=msg["content"]))

        invoke_payload = {
            "chat_history_for_prompt": formatted_history_for_prompt,
            "current_question_for_prompt": question
        }
        print("Invoking general chat chain (async)...")
        answer = await general_chain.ainvoke(invoke_payload)  # Use ainvoke
        print(f"General Answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error generating general answer (async): {e}")
        traceback.print_exc()
        return f"Error: Could not generate general answer. Details: {e}"
