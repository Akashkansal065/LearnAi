import psycopg2
from uuid import UUID, uuid4
from datetime import datetime
from typing import List, Optional, Union, Tuple

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_ollama import OllamaEmbeddings


class PGVectorChatHistory(ChatMessageHistory):
    def __init__(
        self,
        db_url: str,
        session_id: Optional[Union[str, UUID]] = None,
        embedding_model: Optional[OllamaEmbeddings] = None,
        embedding_dim: int = 1024  # match your model
    ):
        self.conn = psycopg2.connect(db_url)
        self.session_id = str(session_id or uuid4())
        self.embedding_model = embedding_model or OllamaEmbeddings(
            model="mxbai-embed-large")
        self.embedding_dim = embedding_dim

        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id UUID NOT NULL,
                    role TEXT CHECK (role IN ('user', 'assistant')),
                    message TEXT NOT NULL,
                    embedding vector(%s),
                    timestamp TIMESTAMP DEFAULT now()
                );
            """, (self.embedding_dim,))
            self.conn.commit()

    def add_user_message(self, message: str) -> None:
        self._insert_message("user", message)

    def add_ai_message(self, message: str) -> None:
        self._insert_message("assistant", message)

    def _insert_message(self, role: str, message: str) -> None:
        embedding = self.embedding_model.embed_query(message)
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_history (session_id, role, message, embedding, timestamp)
                VALUES (%s, %s, %s, %s, %s);
            """, (self.session_id, role, message, embedding, datetime.now()))
            self.conn.commit()

    @property
    def messages(self) -> List[BaseMessage]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT message, role FROM chat_history
                WHERE session_id = %s
                ORDER BY timestamp ASC;
            """, (self.session_id,))
            rows = cur.fetchall()
        return [
            HumanMessage(content=msg) if role == "user" else AIMessage(
                content=msg)
            for msg, role in rows
        ]

    def retrieve_relevant_messages(self, query: str, top_k: int = 5) -> List[Tuple[str, str]]:
        embedding = self.embedding_model.embed_query(query)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT message, role FROM chat_history
                WHERE session_id = %s
                ORDER BY embedding <-> %s
                LIMIT %s;
            """, (self.session_id, embedding, top_k))
            return cur.fetchall()

    def clear(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_history WHERE session_id = %s;", (self.session_id,))
            self.conn.commit()

    def list_sessions(self) -> List[UUID]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT session_id FROM chat_history ORDER BY session_id;")
            rows = cur.fetchall()
        return [row[0] for row in rows]

    def switch_session(self, session_id: Union[str, UUID]) -> None:
        self.session_id = str(session_id)
