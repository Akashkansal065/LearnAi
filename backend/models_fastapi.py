from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class User(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    picture: Optional[str] = None
    is_active: bool = True


class QuestionRequest(BaseModel):
    question: str
    collection_name: str
    llm_model_name: Optional[str] = None  # Allow overriding defaults
    embedding_model_name: Optional[str] = None


class GeneralQuestionRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    llm_model_name: Optional[str] = None


class IndexFileRequest(BaseModel):
    collection_name: str
    processing_mode: str = "text"  # "text" or "ocr"
    llm_model_name: Optional[str] = None
    embedding_model_name: Optional[str] = None
    ocr_model_name: Optional[str] = None


class IndexResponse(BaseModel):
    success: bool
    message: str
    collection_name: Optional[str] = None
    processed_files_count: Optional[int] = None


class AnswerResponse(BaseModel):
    answer: str


class CollectionsListResponse(BaseModel):
    collections: List[str]


class ModelsListResponse(BaseModel):
    models: List[str]


class HealthCheckResponse(BaseModel):
    status: str
