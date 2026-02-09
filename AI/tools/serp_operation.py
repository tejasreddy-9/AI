import json
from serpapi import GoogleSearch
from agno.tools.toolkit import Toolkit
from agno.utils.log import log_info

class SerpTool(Toolkit):
    def __init__(self, api_key):
        super().__init__(name="SerpTool")
        self.api_key = api_key
        self.register(self.search_query)

    def search_query(self, query: str, num_results: int = 10) -> str:
        """
        Searching Google and returning the results.
        Args:
            query(str): search query
            num_results(int): number of results to return, default is 10.
        Returns:
            str: returning the results for search query.
        """
        log_info(f"Searching Google for: {query}")
        params = {
            "q": query,
            "engine": "google",
            "location": "India, Telangana, Hyderabad.",
            "api_key": self.api_key,
            "num": num_results
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        return json.dumps(results)