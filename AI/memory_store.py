# memory_store.py
# -------------------------------------------------------------
# Stores and retrieves chat messages with semantic embeddings
# -------------------------------------------------------------
import hashlib
import os
from typing import List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.inspection import inspect

from agno.knowledge.document import Document
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.embedder.base import Embedder

from nomic_ai import NomicAIEmbedder  # reuse the same embedder

class MemoryStore:
    """
    A tiny wrapper around the existing PostgreSQL+pgVector table –
    `chat_memory`.  Each row stores the full text of a chat turn
    (role + content) and its vector representation.
    """

    def __init__(
        self,
        db_url: str = "postgresql://ai:ai@pgvector:5432/ai",
        table_name: str = "ChatMemory",
        embedder: Optional[Embedder] = None,
    ):
        self.db_url = db_url
        self.table_name = table_name
        self.embedder = embedder or NomicAIEmbedder()

        # create engine + DB session
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)

        # vector extension
        self.vector_db = PgVector(
            table_name=self.table_name,
            db_url=self.db_url,
            embedder=self.embedder,
            search_type=SearchType.vector,
        )

    # ----------------------------------------------------------------
    #  Basic helper – create the vector DB table on first run
    # ----------------------------------------------------------------
    def init_table(self):
        if not inspect(self.engine).has_table(self.table_name):
            with self.Session() as session, session.begin():
                session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                # The underlying PgVector class creates the actual table.
                self.vector_db.create()

    # ----------------------------------------------------------------
    #  Store a message (role + content) as a Document
    # ----------------------------------------------------------------
    def add_message(self, role: str, content: str, batch_size: int = 200):
        self.init_table()
        doc = Document(content=f"{role}: {content}")
        self.vector_db.upsert(documents=[doc])

    # ----------------------------------------------------------------
    #  Return the top‑k relevant memories for a prompt
    # ----------------------------------------------------------------
    def recall(self, query: str, k: int = 5) -> List[str]:
        self.init_table()
        docs = self.vector_db.search(query=query, k=k)
        return [doc.content for doc in docs]
