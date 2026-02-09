from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai import LLMConfig
import os
import asyncio
from agno.tools.toolkit import Toolkit
from agent_knowledge_base import KnowledgeBaseOperation

class CrawlTool(Toolkit):
    def __init__(self):
        super().__init__(name="CrawlTool")
        self.kb = KnowledgeBaseOperation()
        self.register(self.scrap)

    async def main(self, url: str, provider: str) -> str:
        """
        Extract the data from the website using the url.
        Args:
            url(str): scrap the data using the url.
            provider(str): Extract the data using different urls
        Returns:
            str: return the response in the string format.
        """
        browser_conf = BrowserConfig(headless=True)
        conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=LLMExtractionStrategy(
                llm_config=LLMConfig(
                    provider="Ollama/llama3.2",
                    api_token=None
                )
            )
        )

        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(
                url=url,
                config=conf
            )
        
        print(result.markdown)
        content = result.markdown
        return content
    
    def scrap(self, url: str, provider: str) -> str:
        """
        Scrap the data for the given website url.
        Args:
            url(str): Extract the data using the url.
            provider(str): using the different models to extract the data from the website.
        Returns:
            str: return the response for the given url.
        """
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(self.main(url, provider))
            return str(result)
        finally:
            loop.close()