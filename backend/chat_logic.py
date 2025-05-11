import os
import traceback
import asyncio
from typing import Dict, List
import ollama  # For listing models

# Langchain Core/Community imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

# Ollama/Chroma imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient
from ollama_ocr import OCRProcessor
# --- Default Model Constants ---
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = 'llama3:8b'
DEFAULT_OCR_MODEL_NAME = 'llava:latest'

# --- Import your custom OCR Processor ---
try:
    # If ollama_ocr.py is in the same directory (backend/)
    OCR_PROCESSOR_AVAILABLE = True
    print("Successfully imported OCRProcessor from ollama_ocr.py")
except ImportError:
    print("Warning: Could not import OCRProcessor from backend.ollama_ocr.py. Custom OCR mode will use a dummy.")
    OCR_PROCESSOR_AVAILABLE = False

    class OCRProcessor:  # Dummy class if import fails
        def __init__(self, model_name):
            self.model_name = model_name
            print(f"Dummy OCRProcessor initialized with {model_name}")

        async def process_image(self, image_path):  # Made async for consistency
            print(
                f"Dummy OCRProcessor called for {image_path}. Returning no text.")
            await asyncio.sleep(0)  # Simulate async
            return ""

CHROMA_STORE_PATH = './chroma_store'  # Ensure this path is writable

# Create Chroma directory if it doesn't exist
if not os.path.exists(CHROMA_STORE_PATH):
    try:
        os.makedirs(CHROMA_STORE_PATH)
        print(f"Created Chroma directory: {CHROMA_STORE_PATH}")
    except OSError as e:
        print(f"Error creating Chroma directory {CHROMA_STORE_PATH}: {e}")


# --- Helper functions (can be async if they do I/O) ---
async def get_ollama_models_list_async():
    try:
        models_data = await asyncio.to_thread(ollama.list)
        print(f"Fetched {len(models_data.models)} models from Ollama.")
        return [model.model for model in models_data.models]
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

# --- Main Chat Logic (can be a class or functions) ---


async def index_document_async(
    file_path: str,
    file_name: str,  # Original file name for metadata
    collection_name: str,
    processing_mode: str = "text",
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ocr_model_name: str = DEFAULT_OCR_MODEL_NAME
):
    print(f"\n--- Processing document (async): {file_name} ---")
    print(
        f"Target Collection: {collection_name}, Mode: {processing_mode}, Embedding: {embedding_model_name}, OCR: {ocr_model_name}")
    documents = []
    file_extension = os.path.splitext(file_name)[1].lower()

    try:
        if processing_mode == "text":
            if file_extension == ".pdf":
                print("Loader: PyPDFLoader (text mode)")
                loader = PyPDFLoader(file_path)
                # PyPDFLoader.load is sync
                documents = await asyncio.to_thread(loader.load)
            elif file_extension == ".txt":
                print("Loader: TextLoader")
                loader = TextLoader(file_path, encoding='utf-8')
                # TextLoader.load is sync
                documents = await asyncio.to_thread(loader.load)
            else:
                return False, "Unsupported file type for text mode."
        elif processing_mode == "ocr":
            if file_extension == ".pdf":
                print(f"Loader: Custom OCRProcessor (Model: {ocr_model_name})")
                if not OCR_PROCESSOR_AVAILABLE:
                    return False, "OCR Processor not available."
                try:
                    processor = OCRProcessor(model_name=ocr_model_name)
                    extracted_text = await processor.process_image(image_path=file_path)
                    print(
                        f"OCRProcessor completed. Text length: {len(extracted_text)}")
                    if extracted_text:
                        documents = [Document(page_content=extracted_text, metadata={
                                              "source": file_name, "processing_mode": "ocr"})]
                    else:
                        documents = []
                except Exception as ocr_error:
                    print(f"Error during custom OCR processing: {ocr_error}")
                    traceback.print_exc()
                    return False, f"OCR processing error: {ocr_error}"
            else:
                return False, "OCR mode is only supported for PDF files."
        else:
            return False, "Invalid processing mode."

        if not documents:
            print("No content loaded/extracted.")
            return False, "No content extracted from document."
        print(f"Loaded {len(documents)} Langchain Document object(s).")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        chunks = await asyncio.to_thread(text_splitter.split_documents, documents)
        if not chunks:
            print("Failed to split document into chunks.")
            return False, "Failed to split document."
        print(f"Split into {len(chunks)} chunks.")

        # Initialize embeddings model here
        embeddings = OllamaEmbeddings(model=embedding_model_name)
        print(f"Using embedding model: {embedding_model_name}")

        # Chroma.from_documents is synchronous
        await asyncio.to_thread(
            Chroma.from_documents,
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=CHROMA_STORE_PATH
        )
        print(
            f"Successfully added chunks from '{file_name}' to collection '{collection_name}'.")
        return True, f"Successfully processed '{file_name}' into collection '{collection_name}'."

    except Exception as e:
        print(f"Error processing document '{file_name}': {e}")
        traceback.print_exc()
        return False, f"Error processing document: {e}"


async def get_rag_answer_async(
    question: str,
    collection_name: str,
    llm_model_name: str = DEFAULT_LLM_MODEL,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
):
    print(f"\n--- Generating RAG answer (async) ---")
    print(
        f"Question: '{question}', Collection: '{collection_name}', LLM: {llm_model_name}, Embedding: {embedding_model_name}")

    if not collection_name:
        return "Error: No collection selected for RAG."

    try:
        llm = ChatOllama(model=llm_model_name, temperature=0.1)
        embeddings = OllamaEmbeddings(model=embedding_model_name)

        vectorstore = await asyncio.to_thread(
            Chroma,  # Chroma constructor is sync
            persist_directory=CHROMA_STORE_PATH,
            embedding_function=embeddings,
            collection_name=collection_name
        )

        # _collection.count() is sync
        count = await asyncio.to_thread(vectorstore._collection.count)
        if count == 0:
            return f"Error: Collection '{collection_name}' is empty or does not exist."
        print(
            f"Loaded collection '{collection_name}' with {count} embedded chunks.")

        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

        template = """You are an assistant for question-answering tasks based ONLY on the following context.
If you don't know the answer from the context, say that you don't know.
Do not use outside knowledge. Keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("Invoking RAG chain (async)...")
        # Langchain's .ainvoke for async
        answer = await rag_chain.ainvoke(question)
        print(f"RAG Answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error in RAG (async): {e}")
        traceback.print_exc()
        return f"Error: RAG failed. {e}"


async def get_general_answer_async(
    question: str,
    chat_history_list: List[Dict[str, str]],
    llm_model_name: str = DEFAULT_LLM_MODEL
):
    print(f"\n--- Generating General answer (async) ---")
    print(f"Current Question: '{question}', LLM: {llm_model_name}")
    print(f"Incoming History Length: {len(chat_history_list)}")

    try:
        llm = ChatOllama(model=llm_model_name, temperature=0.1)
        general_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer the user's question based on the conversation history if relevant."),
            MessagesPlaceholder(variable_name="chat_history_for_prompt"),
            ("human", "{current_question_for_prompt}")
        ])

        general_chain = general_prompt | llm | StrOutputParser()

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
        # Langchain's .ainvoke for async
        answer = await general_chain.ainvoke(invoke_payload)
        print(f"General Answer: {answer}")
        return answer

    except Exception as e:
        print(f"Error generating general answer (async): {e}")
        traceback.print_exc()
        return f"Error: Could not generate general answer. Details: {e}"
