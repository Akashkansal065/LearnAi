# Chatapp_v3.py
import os
import traceback
import ollama  # For listing models

# Langchain Core/Community imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Ensure AIMessage is imported
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

# Ollama/Chroma imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient

# --- Import your custom OCR Processor ---
try:
    from ollama_ocr import OCRProcessor  # Assuming this file exists
    OCR_PROCESSOR_AVAILABLE = True
except ImportError:
    print("Warning: Could not import OCRProcessor from ollama_ocr.py. Custom OCR mode will use a dummy.")
    OCR_PROCESSOR_AVAILABLE = False

    class OCRProcessor:  # Dummy class if import fails
        def __init__(self, model_name): self.model_name = model_name; print(
            f"Dummy OCRProcessor initialized with {model_name}")
        def process_image(self, image_path): print(
            f"Dummy OCRProcessor called for {image_path}. Returning no text."); return ""


# --- Default Model Constants (can be overridden by Streamlit selections) ---
# Changed from snowflake for broader availability
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = 'llama3:8b'  # Changed to a common llama3 variant
DEFAULT_OCR_MODEL_NAME = 'llava:latest'  # Default for OCR (multimodal)


def get_ollama_models_list():
    """Fetches a list of model names available in the local Ollama instance."""
    try:
        models_data = ollama.list()
        return [model.model for model in models_data.models]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}. Returning empty list.")
        return []


class Chat:
    def __init__(self, vector_store_dir,
                 llm_model_name: str = DEFAULT_LLM_MODEL,
                 embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
                 ocr_model_name: str = DEFAULT_OCR_MODEL_NAME):

        self.vector_store_dir = vector_store_dir
        self.CHROMA_PATH = vector_store_dir  # Store the path for use in methods

        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.ocr_model_name = ocr_model_name

        print(f"Chat instance initializing with:")
        print(f"  LLM Model: {self.llm_model_name}")
        print(f"  Embedding Model: {self.embedding_model_name}")
        print(f"  OCR Model: {self.ocr_model_name}")
        print(f"  ChromaDB Path: {self.CHROMA_PATH}")

        try:
            self.llm = ChatOllama(model=self.llm_model_name, temperature=0.1)
            print(f"Successfully initialized LLM: {self.llm_model_name}")
        except Exception as e:
            print(
                f"ERROR initializing LLM '{self.llm_model_name}': {e}. Backend calls may fail.")
            self.llm = None  # Set to None if initialization fails

        if not os.path.exists(self.CHROMA_PATH):
            try:
                os.makedirs(self.CHROMA_PATH)
                print(f"Created Chroma directory: {self.CHROMA_PATH}")
            except OSError as e:
                print(
                    f"Error creating Chroma directory {self.CHROMA_PATH}: {e}")

    def list_chroma_collections(self):
        try:
            client = PersistentClient(path=self.CHROMA_PATH)
            collections = client.list_collections()
            return collections
        except Exception as e:
            print(
                f"Error listing Chroma collections from {self.CHROMA_PATH}: {e}")
            return []

    def list_chroma_collections(self):
        client = PersistentClient(path=self.vector_store_dir)
        collections = client.list_collections()
        return collections

    def index_pdf(self, file_path, processing_mode: str = "text", collection_name='all'):
        print(f"\n--- Processing document: {os.path.basename(file_path)} ---")
        print(f"Target Collection: {collection_name}, Mode: {processing_mode}")
        documents = []
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if processing_mode == "text":
                # ... (text loading logic as before)
                if file_extension == ".pdf":
                    print("Loader: PyPDFLoader (text mode)")
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                elif file_extension == ".txt":
                    print("Loader: TextLoader")
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
                else:
                    return False
            elif processing_mode == "ocr":
                if file_extension == ".pdf":
                    # Use instance model
                    print(
                        f"Loader: Custom OCRProcessor (Model: {self.ocr_model_name})")
                    if not OCR_PROCESSOR_AVAILABLE:
                        return False  # Handled by dummy if needed
                    try:
                        processor = OCRProcessor(
                            model_name=self.ocr_model_name)  # Use instance model
                        extracted_text = processor.process_image(
                            image_path=file_path)
                        print(
                            f"OCRProcessor completed. Text length: {len(extracted_text)}")
                        if extracted_text:
                            documents = [Document(page_content=extracted_text, metadata={
                                                  "source": os.path.basename(file_path), "processing_mode": "ocr"})]
                        else:
                            documents = []
                    except Exception as ocr_error:
                        print(
                            f"Error during custom OCR processing: {ocr_error}")
                        traceback.print_exc()
                        return False
                else:
                    return False  # OCR only for PDF
            else:
                return False  # Invalid mode

            if not documents:
                print("No content loaded/extracted.")
                return False
            print(f"Loaded {len(documents)} Langchain Document object(s).")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            if not chunks:
                print("Failed to split document.")
                return False
            print(f"Split into {len(chunks)} chunks.")

            embeddings = OllamaEmbeddings(
                model=self.embedding_model_name)  # Use instance model
            print(f"Using embedding model: {self.embedding_model_name}")

            Chroma.from_documents(
                documents=chunks, embedding=embeddings,
                collection_name=collection_name, persist_directory=self.CHROMA_PATH
            )
            print(
                f"Successfully added chunks from '{os.path.basename(file_path)}' to collection '{collection_name}'.")
            return True

        except Exception as e:
            print(
                f"Error processing document '{os.path.basename(file_path)}': {e}")
            traceback.print_exc()
            return False

    def get_answer(self, question: str, collection_name: str):
        print(f"\n--- Generating RAG answer ---")
        if not self.llm:
            return "Error: LLM not initialized. Cannot generate answer."
        print(f"Question: '{question}', Collection: '{collection_name}'")
        if not collection_name:
            return "Error: No collection selected for RAG."

        try:
            embeddings = OllamaEmbeddings(
                model=self.embedding_model_name)  # Use instance model
            vectorstore = Chroma(
                persist_directory=self.CHROMA_PATH,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            count = vectorstore._collection.count()
            if count == 0:
                return f"Error: Collection '{collection_name}' is empty."
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
            rag_chain = ({"context": retriever, "question": RunnablePassthrough(
            )} | prompt | self.llm | StrOutputParser())
            print("Invoking RAG chain...")
            answer = rag_chain.invoke(question)
            print(f"RAG Answer: {answer}")
            return answer
        except Exception as e:
            print(f"Error in RAG: {e}")
            traceback.print_exc()
            return f"Error: RAG failed. {e}"

    # chat_history_list is the history *before* the current question
    def get_general_answer(self, question: str, chat_history_list: list):
        print(f"\n--- Generating General answer ---")
        if not self.llm:
            return "Error: LLM not initialized. Cannot generate answer."
        print(f"Current Question: '{question}'")
        print(f"Incoming History Length: {len(chat_history_list)}")

        try:
            # Prompt with history placeholder
            general_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Answer the user's question based on the conversation history if relevant."),
                # Use a distinct name for clarity
                MessagesPlaceholder(variable_name="chat_history_for_prompt"),
                # The current question
                ("human", "{current_question_for_prompt}")
            ])

            # Chain (uses self.llm initialized in __init__)
            general_chain = (general_prompt | self.llm | StrOutputParser())

            # Format history for MessagesPlaceholder
            # This list should contain all prior user and assistant messages
            formatted_history_for_prompt = []
            for msg in chat_history_list:  # This is the history *before* the current question
                if msg["role"] == "user":
                    formatted_history_for_prompt.append(
                        HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history_for_prompt.append(
                        AIMessage(content=msg["content"]))

            # dafaa
            invoke_payload = {
                "chat_history_for_prompt": formatted_history_for_prompt,
                "current_question_for_prompt": question
            }
            print(f"\n[DEBUG] Invoking general_chain with payload:")
            print(
                f"  Current Question: {invoke_payload['current_question_for_prompt']}")
            print(
                f"  Formatted History for Prompt ({len(invoke_payload['chat_history_for_prompt'])} messages):")
            for i, hist_msg in enumerate(invoke_payload['chat_history_for_prompt']):
                # Print first 100 chars
                print(
                    f"    {i}: Type={type(hist_msg).__name__}, Content='{hist_msg.content[:100]}...'")

            # This will show what messages are actually being constructed from the template and inputs
            try:
                # Get the fully resolved messages that will be sent to the LLM
                resolved_messages = general_prompt.format_messages(
                    **invoke_payload)
                print("\n[DEBUG] Resolved messages to be sent to LLM:")
                for i, res_msg in enumerate(resolved_messages):
                    print(
                        f"    {i}: Type={type(res_msg).__name__}, Role='{res_msg.type if hasattr(res_msg, 'type') else 'N/A'}', Content='{res_msg.content[:200]}...'")
            except Exception as e_res:
                print(
                    f"[DEBUG] Error resolving messages for printing: {e_res}")

            # Invoke chain
            # The current question is passed separately and handled by the "human" part of the prompt
            print("Invoking general chat chain...")
            answer = general_chain.invoke({
                "chat_history_for_prompt": formatted_history_for_prompt,
                "current_question_for_prompt": question  # The new question from the user
            })
            print(f"General Answer: {answer}")
            return answer

        except Exception as e:
            print(f"Error generating general answer: {e}")
            traceback.print_exc()
            return f"Error: Could not generate general answer. Details: {e}"
