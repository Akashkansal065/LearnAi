from typing import Optional
from fastapi import Form
import os
import tempfile
import shutil
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
# For potential CORS issues with Streamlit
from fastapi.middleware.cors import CORSMiddleware
# Required for Authlib's OAuth state
from starlette.middleware.sessions import SessionMiddleware
import uvicorn

# Import Pydantic models and logic functions
from models_fastapi import (
    QuestionRequest, GeneralQuestionRequest, IndexFileRequest,
    AnswerResponse, IndexResponse, CollectionsListResponse, ModelsListResponse,
    HealthCheckResponse, User
)
from chat_logic import (
    get_ollama_models_list_async, list_chroma_collections_async,
    index_document_async, get_rag_answer_async, get_general_answer_async,
    DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL, DEFAULT_OCR_MODEL_NAME
)
# Import auth router and dependencies
from auth_fastapi import router as auth_router, get_current_active_user

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Chat Application API")

# Middleware
# Add SessionMiddleware for OAuth state handling (Authlib requires it)
# Ensure JWT_SECRET_KEY is set for SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv(
    "JWT_SECRET_KEY", "your-fallback-secret-key-for-session"))

# CORS (Cross-Origin Resource Sharing)
# Adjust origins as necessary for your Streamlit app's deployment
origins = [
    os.getenv("STREAMLIT_SERVER_ADDRESS",
              "http://localhost:8501"),  # Streamlit default
    "http://127.0.0.1:8501",
    # Add other origins if needed, e.g., your production Streamlit URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Important for cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the authentication router
app.include_router(auth_router, tags=["Authentication"])


@app.get("/", response_model=HealthCheckResponse, tags=["General"])
async def root():
    return HealthCheckResponse(status="API is running")


@app.get("/ollama-models", response_model=ModelsListResponse, tags=["Models & Collections"])
async def get_ollama_models(current_user: User = Depends(get_current_active_user)):
    models = await get_ollama_models_list_async()
    return ModelsListResponse(models=models)


@app.get("/chroma-collections", response_model=CollectionsListResponse, tags=["Models & Collections"])
async def get_chroma_collections(current_user: User = Depends(get_current_active_user)):
    print(current_user)
    collections = await list_chroma_collections_async()
    return CollectionsListResponse(collections=collections)


@app.post("/index-file/", response_model=IndexResponse, tags=["Chat & Indexing"])
async def index_file_endpoint(
    collection_name: str = Form(...),
    processing_mode: str = Form(...),
    llm_model_name: Optional[str] = Form(None),
    embedding_model_name: Optional[str] = Form(None),
    ocr_model_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, file.filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            success, message = await index_document_async(
                file_path=temp_file_path,
                file_name=file.filename,
                collection_name=collection_name,
                processing_mode=processing_mode,
                embedding_model_name=embedding_model_name or DEFAULT_EMBEDDING_MODEL,
                ocr_model_name=ocr_model_name or DEFAULT_OCR_MODEL_NAME
            )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error during file processing: {str(e)}")
    finally:
        await file.close()

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return IndexResponse(
        success=True,
        message=message,
        collection_name=collection_name,
        processed_files_count=1
    )


@app.post("/get-rag-answer/", response_model=AnswerResponse, tags=["Chat & Indexing"])
async def get_rag_answer_endpoint(
    request_data: QuestionRequest,
    current_user: User = Depends(get_current_active_user)
):
    answer = await get_rag_answer_async(
        question=request_data.question,
        collection_name=request_data.collection_name,
        llm_model_name=request_data.llm_model_name or DEFAULT_LLM_MODEL,
        embedding_model_name=request_data.embedding_model_name or DEFAULT_EMBEDDING_MODEL
    )
    if answer.startswith("Error:"):
        raise HTTPException(status_code=400, detail=answer)
    return AnswerResponse(answer=answer)


@app.post("/get-general-answer/", response_model=AnswerResponse, tags=["Chat & Indexing"])
async def get_general_answer_endpoint(
    request_data: GeneralQuestionRequest,
    current_user: User = Depends(get_current_active_user)
):
    answer = await get_general_answer_async(
        question=request_data.question,
        chat_history_list=request_data.chat_history,
        llm_model_name=request_data.llm_model_name or DEFAULT_LLM_MODEL
    )
    if answer.startswith("Error:"):
        raise HTTPException(status_code=400, detail=answer)
    return AnswerResponse(answer=answer)

# To run this app (save as main_fastapi.py in backend/):
# Ensure your .env file is in the parent directory or backend/
# From the parent directory: uvicorn backend.main_fastapi:app --reload --port 8000
# Or from the backend/ directory: uvicorn main_fastapi:app --reload --port 8000

if __name__ == "__main__":
    #     # gunicorn.run("main:app", host="0.0.0.0", port=8087, reload=True)
    #     #     #     # print_hi("PyCharm")
    #     #     #     #
    #     #     #     # uvicorn main:app --reload --port 8080 --host 0.0.0.0 --ssl-keyfile "/home/akash.kansal/documents/github/dp-core-automation/certkey.pem" --ssl-certfile "/home/akash.kansal/documents/github/dp-core-automation/cert.pem"
    # uvicorn.run("main:app", host="0.0.0.0", port=8088, reload=True)  # , workers=4)
    uvicorn.run("main_fastapi:app", host="0.0.0.0", port=8080, reload=True,
                # ssl_keyfile="./certkey.pem",
                # ssl_certfile="./cert.pem"
                )  # , workers=4)
