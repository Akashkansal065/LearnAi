import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
import traceback
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from chromadb import PersistentClient

CHROMA_PATH = "./chroma_store"
EMBEDDING_MODEL = "snowflake-arctic-embed2"
LLM_MODEL = 'gemma3:12b'


class Chat:
    def __init__(self, vector_store_dir):
        self.vector_store_dir = vector_store_dir
        self.chat_history = []
        self.OUTPUT_DIR = "./output_md"
        self.chat_history_manager = StreamlitChatMessageHistory()  # Manages chat history

    def list_chroma_collections(self):
        client = PersistentClient(path=self.vector_store_dir)
        collections = client.list_collections()
        return collections

    def create_prompt_template(self):
        """Create a ChatPromptTemplate with message placeholders."""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that answers questions based on the uploaded documents."),
            ("user", "{user_input}")
        ])
        return prompt_template

    def get_default_collection(self):
        collections = self.list_collections()
        if collections:
            return collections[0]
        return None

    def get_collection_metadata(self, collection_name):
        try:
            collection = self.db.get_collection(name=collection_name)
            return collection.metadata
        except Exception as e:
            print(f"Error getting metadata for {collection_name}: {e}")
            return {}

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
                elif file_extension == ".txt":
                    print("Using TextLoader...")
                    # Specify encoding
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
                else:
                    print(
                        f"Error: Unsupported file type '{file_extension}' for text mode.")
                    return False
            elif processing_mode == "ocr":
                if file_extension == ".pdf":
                    print("Using UnstructuredPDFLoader (OCR mode)...")
                    # Requires tesseract to be installed and potentially in PATH
                    # "hi_res" strategy uses detectron2 if available, falls back to Tesseract
                    # "ocr_only" forces OCR extraction
                    loader = UnstructuredPDFLoader(
                        file_path,
                        mode="single",  # Process pages individually
                        strategy="hi_res"  # or "ocr_only" if needed
                    )
                    documents = loader.load()
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
        # if processing_mode == "text":
        #     loader = PyPDFLoader(file_path)
        #     documents = loader.load()
        # else:
        #     processor = OCRProcessor(model_name='llama3.2-vision')
        #     text = processor.process_image(
        #         image_path=file_path, format_type="markdown")
        #     output_path = pathlib.Path(self.OUTPUT_DIR) / f"temp_output.md"
        #     output_path.write_text(text, encoding="utf-8")
        #     documents = [Document(page_content=text, metadata={
        #                           "source": str(file_path)})]

        # splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=500, chunk_overlap=50)
        # docs = splitter.split_documents(documents)
        # doc_ids = [str(uuid.uuid4()) for _ in range(len(docs))]

        # embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2")
        # db = Chroma.from_documents(
        #     documents=docs,
        #     embedding=embeddings,
        #     ids=doc_ids,
        #     persist_directory=self.vector_store_dir,
        #     collection_name=collection_name
        # )
        # return len(docs)

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
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings, collection_name=collection_name
            )
            # 3. Create Retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",  # Default, can also be "mmr"
                search_kwargs={'k': 3}     # Retrieve top 3 relevant chunks
            )
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
