from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.inspection import inspect
from agno.knowledge.document import Document
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.reader.base import Reader
from agno.utils.log import log_debug, log_info
from nomic_ai import NomicAIEmbedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agno.vectordb.pgvector import PgVector, SearchType
from agno.vectordb import VectorDb
from typing import Optional, List
import hashlib

class KnowledgeBaseOperation():
    def __init__(
        self,
        reader: Optional[Reader] = PDFReader(),
        vector_db: Optional[VectorDb] = None,
        num_documents: int = 3,
        search_type: SearchType = SearchType.vector
    ):
        self.reader = reader
        self.num_documents = num_documents
        self.table_name = 'Agent_knowledge'
        self.db_url = 'postgresql://ai:ai@pgvector:5432/ai'
        self.schema = None

        self.db_engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.db_engine)

        self.embedder = NomicAIEmbedder()

        self.vector_db = PgVector(
            table_name=self.table_name,
            db_url=self.db_url,
            embedder=NomicAIEmbedder()
        )

    def chunk(self, content: str, chunk_size: int = 2000, chunk_overlap: int = 0):
        """
        Divide the content into small paragraphs.
        Args:
            content(str): Data that we are using in PDFs, URLs
            chunk_size(int): convert the data into words using the size.
            chunk_overlap(int): Does not repeat again.
        """
        text_splitters = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return text_splitters.split_text(content)
        
    def _embeddings_(self, text: str):
        """
        Convert the data into numerical representation.
        Args:
            text(str): we are taking the text in the form PDFs, Urls.
        """
        return self.embedder.get_embedding(text)
    
    def table_exists(self):
        """Checking whether the table exists in database or not."""
        log_debug(f"check the table {self.table_name} is already exists are not")
        return inspect(self.db_engine).has_table(self.table_name, schema=self.schema)
    
    def create(self):
        """Creating the table."""
        table_exists = self.table_exists()
        print("Table exists:", table_exists)
        
        if not table_exists:
            with self.Session() as session, session.begin():
                log_debug("creating extension: vector")
                session.execute(text("create extension if not exists vector"))
                if self.schema:
                    log_debug(f"Creating schema: {self.schema}")
                    session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema};"))
                
                self.vector_db.create()
                log_info(f"Created table {self.table_name} successfully.")

    def insert(self, documents: List[Document], batch_size: int = 200):
        """
        Inserting the data.
        Args:
            documents(list): list of documents
            batch_size(int): Number of documents to insert into the batch.
        """
        for i in range(0, len(documents), batch_size):
            docs = documents[i:i+batch_size]
            log_debug(f"from batch start index {i} to {len(docs)}")
            for doc in docs:
                unique_hash = hashlib.sha256(doc.content.encode("utf-8")).hexdigest()
                self.vector_db.upsert(documents=[doc], content_hash=unique_hash)

    def upsert(self, documents: List[Document], batch_size: int = 2000):
        """
        We are updating and inserting the data.
        Args:
            documents(list): list of documents.
            batch_size(int): Number of documents to update in the each batch.
        """
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i: i + batch_size]
            log_debug(f"From batch start index {i} to {len(batch_docs)}")
            for doc in batch_docs:
                unique_hash = hashlib.sha256(doc.content.encode("utf-8")).hexdigest()
                self.vector_db.upsert(documents=[doc], content_hash=unique_hash)

    def search_query(self, query: str):
        """
        Perform the search based on the query.
        Args:
            query(str): search the query
        """
        response = self.vector_db.search(query=query)
        log_debug(f"Found the {len(response)} matching document.")
        return response
    
    def text_data(self, text: str):
        """
        Loads and store the text data in database.
        Args:
            text(str): input text
        """
        try:
            chunks = self.chunk(text)
            print(f"Chunked into {len(chunks)} parts")

            data = []
            for idx, chunk in enumerate(chunks):
                preview = chunk[:200].replace("\n", " ")
                print(f"Chunk {idx} preview: {preview}...")

                embedding = self._embeddings_(chunk)
                print(f"Embedding type for chunk {idx}: {type(embedding)}; length: {len(embedding) if hasattr(embedding, '__len__') else 'unknown'}")
                
                doc = Document(
                    content=chunk, 
                    embedding=embedding, 
                )
                data.append(doc)

            self.create()
            self.upsert(documents=data)
            log_info(f"Inserted {len(data)} records into knowledge base.")
        except Exception as e:
            log_info(f"Error while storing text data: {e}")
            raise e