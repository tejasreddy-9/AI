import fitz
import os
from agent_knowledge_base import KnowledgeBaseOperation
from pathlib import Path
from agno.tools.toolkit import Toolkit

class PDFKnowledgeBase(Toolkit):  
    """Uploading the data it is extracted to text format."""
    
    def __init__(self):
        super().__init__(name="PDFKnowledgeBase")
        self.ak_tool = KnowledgeBaseOperation()
        self.loaded_files = set()
        self.register(self.write_and_search)

    def write_and_search(self, pdf_path: str, query: str):
        """
        Extracts the content of a pdf and then search it based on the query provided.
        Args:
            pdf_path (str): the path to the pdf 
            query (str): what to be extracted
        Returns:
            str: Search results
        """
        print("******************************** query:", query)
        
        # Use provided pdf_path first, fall back to self.file_path
        if not pdf_path and hasattr(self, 'file_path'):
            pdf_path = self.file_path
        
        if not pdf_path or not os.path.exists(pdf_path):
            return "No PDF file loaded or file doesn't exist. Upload a PDF first."
        
        print(f"Using file_path: {pdf_path}")
        
        self.pdf_upload(pdf_path)
        results = self.search(query)  
        return "\n\n".join(results) if results else "No results found."

    def extract_text_from_pdf(self, pdf_path) -> str:
        """Extract text from a single PDF file using PyMuPDF."""
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def extract_text_from_txt(self, txt_path) -> str:
        """Extract text from a single .txt file."""
        txt_path = str(txt_path)
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def pdf_upload(self, path: str):
        """
        To upload the document in the form of pdf.
        Args:
            path(str): path is used to upload the document.
        """
        print(f"Loading PDF from path: {path}")
        
        if not os.path.exists(path):
            print(f"ERROR: File not found at path: {path}")
            return
        
        print(f"File exists at: {path}") 
        
        self.ak_tool.create()
        self.loaded_files.clear()
        pdf_path = Path(path)
        documents = []
        
        if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
            print(f"Loading PDF (per-page): {path}")
            with fitz.open(path) as doc:
                for pnum, page in enumerate(doc):
                    page_text = page.get_text()
                    if not page_text or not page_text.strip():
                        continue
                    page_key = f"{path}::page::{pnum}"
                    if page_key in self.loaded_files:
                        continue
                    documents.append(page_text)
                    self.loaded_files.add(page_key)

        elif pdf_path.is_dir():
            for pdf in pdf_path.glob("**/*.pdf"):
                if pdf.suffix.lower() == ".pdf":
                    pdf_str = str(pdf)
                    print(f"Loading PDF from dir: {pdf_str}")
                    with fitz.open(pdf_str) as doc:
                        for pnum, page in enumerate(doc):
                            page_text = page.get_text()
                            if not page_text or not page_text.strip():
                                continue
                            page_key = f"{pdf_str}::page::{pnum}"
                            if page_key in self.loaded_files:
                                continue
                            documents.append(page_text)
                            self.loaded_files.add(page_key)
        
        print(f"Total documents to process: {len(documents)}")  
        
        combined_text = "\n\n".join(documents)
        self.ak_tool.text_data(combined_text)

        print(f"✅ Successfully processed {len(documents)} documents")

    def search(self, query: str):
        """
        Search the query in the database.
        Args:
            query(str): search the query
        Returns:
            list: List of matching content
        """
        response = self.ak_tool.vector_db.search(query=query)
        resp = [doc.content for doc in response]
        return resp