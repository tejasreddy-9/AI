# memory_tool.py
# -------------------------------------------------------------
# A LangChain Agent tool that lets the agent query the conversation history.
# -------------------------------------------------------------
from agno.tools.toolkit import Toolkit
from memory_store import MemoryStore

class MemoryTool(Toolkit):
    """
    A simple tool that can be invoked by an Agent to access recent
    or semantically‑relevant chat history.
    """
    name = "MemoryTool"

    def __init__(self):
        super().__init__(name=self.name)
        self.memory = MemoryStore()
        self.register(self.recall)

    def recall(self, query: str) -> str:
        """
        Search the memory DB and return a text block that contains the
        top‑k most relevant previously stored turns.
        """
        results = self.memory.recall(query=query)
        return "\n\n".join(results) if results else "No relevant memory found."
