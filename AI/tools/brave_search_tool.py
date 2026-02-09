import json
from langchain_community.document_loaders import BraveSearchLoader
from agno.tools.toolkit import Toolkit
import os
from dotenv import load_dotenv

load_dotenv()

class BraveSearch(Toolkit):
    """Search query results from Brave."""
    
    def __init__(self, api_key, num_results=10):
        super().__init__(name="BraveSearch")
        self.api_key = api_key
        self.num_results = num_results
        self.register(self.search)

    def search(self, query: str) -> str:
        """
        Search the results for the given query.
        Args:
            query(str): search query
        Returns:
            Returning the results from search query.
        """
        try:
            loader = BraveSearchLoader(
                query=query,
                api_key=self.api_key,
                search_kwargs={"count": self.num_results}
            )
            docs = loader.load()
            content = [doc.page_content for doc in docs]
            links = [doc.metadata for doc in docs]
            combined = [{"content": c, "link": l} for c, l in zip(content, links)]
            result = json.dumps(combined)
            return result
        except Exception as e:
            return f"Error searching for the query {query}: {e}"