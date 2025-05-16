import os
import pathlib
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
import traceback
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from chromadb import PersistentClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from ollama_ocr import OCRProcessor
from langchain_core.documents import Document


CHROMA_PATH = "./chroma_store"
EMBEDDING_MODEL = "snowflake-arctic-embed2"
LLM_MODEL = 'gemma3:12b'
OCR_MODEL_NAME = 'llama3.2-vision'
# OCR_MODEL_NAME = 'llava:13b'


class Chat:
    def __init__(self, vector_store_dir):
        self.vector_store_dir = vector_store_dir
        self.chat_history = []
        self.OUTPUT_DIR = "./output_md"
        self.chat_history_manager = StreamlitChatMessageHistory()
        global CHROMA_PATH
        CHROMA_PATH = vector_store_dir  # Use the path provided during instantiation
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
        self.output_dir = "./output_md"
        print(f"Initialized LLM: {LLM_MODEL}")
        # Check if OCR processor can be initialized (optional, depends on OCRProcessor design)
        # if OCR_PROCESSOR_AVAILABLE:
        #     try:
        #         # Test initialization if needed, handle potential errors
        #         _ = OCRProcessor(model_name=OCR_MODEL_NAME)
        #         print(f"OCR Processor using model '{OCR_MODEL_NAME}' seems available.")
        #     except Exception as e:
        #         print(f"Warning: Could not initialize OCRProcessor: {e}")

    def list_chroma_collections(self):
        client = PersistentClient(path=self.vector_store_dir)
        collections = client.list_collections()
        return collections

    # def create_prompt_template(self):
    #     """Create a ChatPromptTemplate with message placeholders."""
    #     prompt_template = ChatPromptTemplate.from_messages([
    #         ("system", "You are an assistant that answers questions based on the uploaded documents."),
    #         ("user", "{user_input}")
    #     ])
    #     return prompt_template

    def index_pdf(self, file_path, processing_mode: str = "text", collection_name='all'):
        """
    Loads, splits, embeds, and stores document content in ChromaDB based on mode.

    Args:
        file_path: Path to the document file (.pdf or .txt).
        collection_name: Name for the ChromaDB collection.
        processing_mode: 'text' for text-based loading, 'ocr' for OCR loading.

    Returns:
        True if processing was successful, False otherwise.
    """
        print(
            f"Processing document: {file_path} into collection: {collection_name} using mode: {processing_mode}")
        documents = []
        file_extension = os.path.splitext(file_path)[1].lower()
        try:
            # 1. Load Document based on mode and extension
            if processing_mode == "text":
                if file_extension == ".pdf":
                    print("Using PyPDFLoader (text mode)...")
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                elif file_extension in [".txt", ".md"]:
                    print("Using TextLoader...")
                    # Specify encoding
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
                else:
                    print(
                        f"Error: Unsupported file type '{file_extension}' for text mode.")
                    return False
            elif processing_mode == "ocr":
                if file_extension.lower() == ".pdf" or file_extension.lower() in [".pdf", ".jpg", ".jpeg", ".png"]:
                    print("Using UnstructuredPDFLoader (OCR mode)...")
                    # Requires tesseract to be installed and potentially in PATH
                    # "hi_res" strategy uses detectron2 if available, falls back to Tesseract
                    # "ocr_only" forces OCR extraction
                    # loader = UnstructuredPDFLoader(
                    #     file_path,
                    #     mode="single",  # Process pages individually
                    #     strategy="hi_res"  # or "ocr_only" if needed
                    # )
                    # documents = loader.load()
                    try:
                        # --- Use Custom OCR Processor ---
                        print(
                            f"Instantiating OCRProcessor with model: {OCR_MODEL_NAME}")
                        processor = OCRProcessor(model_name=OCR_MODEL_NAME)

                        print(f"Processing PDF with OCRProcessor: {file_path}")
                        # Assuming process_image takes the pdf path and returns a single text string
                        extracted_text = processor.process_image(
                            image_path=file_path)
                        print(extracted_text)
                        output_path = pathlib.Path(
                            self.output_dir) / f"{str(int(time.time()))}_output.md"
                        print(output_path)
                        output_path.write_text(
                            extracted_text, encoding="utf-8")
                        print(
                            f"OCRProcessor finished. Extracted text length: {len(extracted_text)}")

                        if extracted_text:
                            # --- Wrap extracted text in Langchain Document object ---
                            documents = [Document(page_content=extracted_text, metadata={
                                                  "source": str(file_path), "processing_mode": "ocr"})]
                        else:
                            print("Warning: OCRProcessor returned empty text.")
                            documents = []  # Ensure documents list is empty if OCR fails

                    except Exception as ocr_error:
                        print(
                            f"Error during custom OCR processing: {ocr_error}")
                        traceback.print_exc()
                        return False  # Fail processing if OCR step fails
                else:
                    print("Error: OCR mode only supports PDF files.")
                    return False
            else:
                print(f"Error: Invalid processing mode '{processing_mode}'.")
                return False

            if not documents:
                print("Error: Could not load any content from the document.")
                return False
            print(f"Loaded {len(documents)} document sections/pages.")

            # 2. Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            if not chunks:
                print("Error: Could not split the document into chunks.")
                return False
            print(f"Split into {len(chunks)} chunks.")

            # 3. Create Embeddings using Ollama
            print(f"Initializing embeddings with model: {EMBEDDING_MODEL}")
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

            # 4. Store in ChromaDB (Persistent)
            print(
                f"Creating/loading ChromaDB collection: {collection_name} at {CHROMA_PATH}")
            # Check if collection exists? Overwrite or append? Overwriting for simplicity now.
            # To delete existing before adding:
            # try:
            #     chroma_client = Chroma(persist_directory=CHROMA_PATH)
            #     chroma_client.delete_collection(name=collection_name)
            #     print(f"Deleted existing collection: {collection_name}")
            # except Exception as e:
            #     print(f"Note: Could not delete collection {collection_name} (may not exist): {e}")

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=CHROMA_PATH
            )

            print(
                f"Successfully processed and stored document in Chroma collection '{collection_name}'.")
            return True

        except ImportError as ie:
            print(
                f"Import Error: {ie}. Did you install required libraries for the chosen mode?")
            if "unstructured" in str(ie).lower():
                print("-> The 'ocr' mode requires 'unstructured' and its dependencies.")
                print("-> Try: pip install \"unstructured[pdf]\"")
            if "detectron2" in str(ie).lower():
                print(
                    "-> 'unstructured' hi_res strategy benefits from 'detectron2'. Check its installation guide.")
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"Error processing document '{file_path}': {e}")
            if "tesseract is not installed or not in your PATH" in str(e).lower():
                print("\n------ TESSERACT OCR ERROR ------")
                print(
                    "Tesseract is required for the 'Contains Images/OCR' mode but was not found.")
                print("Please install Tesseract OCR for your system:")
                print("  - Ubuntu/Debian: sudo apt install tesseract-ocr")
                print("  - macOS: brew install tesseract")
                print(
                    "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
                print(
                    "Ensure the Tesseract installation path is included in your system's PATH environment variable.")
                print("---------------------------------\n")
            traceback.print_exc()
            return False

    def get_answer(self, question: str, collection_name):
        """Retrieves relevant context from ChromaDB and generates an answer using Ollama LLM.
        Args:question: The user's question.collection_name: The ChromaDB collection associated with the relevant PDF.
        Returns:The generated answer string, or an error message."""
        try:
            # 1. Initialize Embeddings (needed for loading the store)
            print(
                f"Initializing embeddings ({EMBEDDING_MODEL}) for retrieval.")
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = Chroma(
                persist_directory='./chroma_store',
                embedding_function=embeddings, collection_name=collection_name
            )
            print(collection_name)
            # 3. Create Retriever
            # retriever = vectorstore.as_retriever(
            #     search_type="similarity",  # Default, can also be "mmr"
            #     search_kwargs={'k': 3}     # Retrieve top 3 relevant chunks
            # )
            retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 6, "fetch_k": 10, "lambda_mult": 0.5}
            )
            print(retriever)

            print("Retriever created.")
            # 4. Define RAG Prompt Template
            template = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question accurately.
            If you don't know the answer based on the context, just say that you don't know.
            Provide a concise answer based *only* on the provided context.

            Context:
            {context}

            Question: {question}

            Answer:"""
            prompt = ChatPromptTemplate.from_template(template)

            # 5. Initialize LLM
            print(f"Initializing LLM: {LLM_MODEL}")
            # Low temp for factual answers
            llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

            # 6. Create RAG Chain using LangChain Expression Language (LCEL)
            rag_chain = (
                # RunnableParallel allows running retriever and question passthrough concurrently
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # 7. Invoke Chain and Get Answer
            print("Invoking RAG chain...")
            answer = rag_chain.invoke(question)
            print(f"Generated answer: {answer}")
            return answer

        except Exception as e:
            print(
                f"Error generating answer for collection: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: Could not generate answer. Details: {e}"

    # --- NEW METHOD for General Chat ---
    def get_general_answer(self, question: str, chat_history_list: list):
        """ (Non-RAG Answer) Generates an answer using only the LLM and chat history. """
        print(f"Generating general answer for: '{question}'")
        try:
            # Basic prompt for general conversation
            # Incorporate chat history using MessagesPlaceholder
            general_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Answer the user's question."),
                # Placeholder for history
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])

            # Create the chain (using the pre-initialized self.llm)
            general_chain = (
                general_prompt
                | self.llm  # Use the initialized LLM
                | StrOutputParser()
            )

            # Invoke the chain with the question and formatted history
            print("Invoking general chat chain...")
            # Format history for MessagesPlaceholder (needs list of BaseMessage objects)
            # Assuming chat_history_list is like [{"role": "user", "content": "..."}, ...]
            formatted_history = []
            for msg in chat_history_list:
                if msg["role"] == "user":
                    formatted_history.append(
                        HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    # Assuming ChatOllama produces AIMessage compatible output
                    # If not, adjust this line based on actual message type
                    formatted_history.append(AIMessage(content=msg["content"]))
                    # formatted_history.append(
                    #     self.llm.get_lc_message_type()(content=msg["content"]))
            formatted_history.append(HumanMessage(content=question))

            answer = general_chain.invoke({
                "question": question,
                "chat_history": formatted_history  # Pass the formatted history
            })
            print(f"Generated general answer: {answer}")
            return answer

        except Exception as e:
            print(f"Error generating general answer: {e}")
            traceback.print_exc()
            return f"Error: Could not generate general answer. Details: {e}"

# You can add a small test block here if needed
# if __name__ == '__main__':
#     print("Running backend checks...")
#     if is_ollama_running():
#         # Add a dummy PDF path and collection name for testing if needed
#         # test_pdf = "path/to/your/test.pdf"
#         # test_collection = "test_pdf_collection"
#         # if os.path.exists(test_pdf):
#         #     process_and_store_pdf(test_pdf, test_collection)
#         #     print(get_answer("What is the main topic?", test_collection))
#         pass
#     else:
#         print("Ollama is not running. Backend functions may fail.")
