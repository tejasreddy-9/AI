# main.py
# --------------------------------------------------------------
# FastAPI + Agno Agent + MemoryTool
# Author: <your name>
# --------------------------------------------------------------

from typing import Dict, Any

import os
import shutil

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

# ───────────────────────────────────────────────────────────────
# Imports from the original code base
# ───────────────────────────────────────────────────────────────
from agno.agent import Agent
from _pdf_ import PDFKnowledgeBase as PDFKB
from langextract_main import get_langextract_tool
from tools.brave_search_tool import BraveSearch
from tools.serp_operation import SerpTool
from tools.crawl_ai import CrawlTool
# ───────────────────────────────────────────────────────────────
# Memory pieces – added in the previous response
# ───────────────────────────────────────────────────────────────
from memory_tool import MemoryTool
# ───────────────────────────────────────────────────────────────
# LLM providers
# ───────────────────────────────────────────────────────────────
from agno.models.mistral import MistralChat
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.models.perplexity import Perplexity
from agno.models.anthropic import Claude
# --------------------------------------------------------------
load_dotenv()

app = FastAPI()

# --------------------------------------------------------------
# Upload folder (used by the contract‑parser tool)
# --------------------------------------------------------------
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploaded_files")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Receive uploaded file and save to shared upload folder."""
    dest_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    await file.close()
    return {"path": dest_path}

# --------------------------------------------------------------
# Prompt model – the body of a `/chat/` POST request
# --------------------------------------------------------------
class Prompt(BaseModel):
    message: str
    provider: str
    api_key: str
    id: str
    tool: str
    tool_config: Dict[str, Any]

@app.get("/")
def read_root():
    return {"Message": "Welcome to the FastAPI Agent"}

# --------------------------------------------------------------
# Helper: create the right LLM instance
# --------------------------------------------------------------
def get_model(provider: str, model_id: str, api_key: str):
    provider = provider.lower()
    if provider == "gemini":
        return Gemini(id=model_id, api_key=api_key)
    if provider == "openai":
        return OpenAIChat(id=model_id, api_key=api_key)
    if provider == "groq":
        return Groq(id=model_id, api_key=api_key)
    if provider == "mistral":
        return MistralChat(id=model_id, api_key=api_key)
    if provider == "perplexity":
        return Perplexity(id=model_id, api_key=api_key)
    if provider == "ollama":
        return Ollama(id=model_id)          # Ollama doesn't use an API key
    if provider == "claude":
        return Claude(id=model_id, api_key=api_key)

    raise ValueError(f"Unsupported provider: {provider}")

# --------------------------------------------------------------
# Memory handling – our custom tool
# --------------------------------------------------------------
memory_tool = MemoryTool()

# Optional helper routes – not strictly needed but handy in dev
@app.post("/memory/add")
def add_to_memory(payload: Dict[str, str]):
    """
    Payload: {"role": "user|assistant", "content": "text"}
    """
    role = payload.get("role")
    content = payload.get("content")
    if not role or not content:
        return {"error": "Both 'role' and 'content' must be supplied"}
    memory_tool.memory.add_message(role, content)
    return {"status": "stored"}

@app.get("/memory/recall")
def recall_memory(query: str, k: int = 5):
    """
    Retrieve the top‑k most relevant memory messages:
    GET /memory/recall?query=...
    """
    return {"results": memory_tool.memory.recall(query, k)}

# --------------------------------------------------------------
# The single chat endpoint – everything runs here
# --------------------------------------------------------------
@app.post("/chat/")
def chat_with_agent(prompt: Prompt):
    """
    Handles the chat request, applies tools and persists the
    conversational history to memory.
    """

    # 1. Start with the base tools that the user requested
    tools = []

    # Ask the user for additional tools
    if prompt.tool == "brave_search" and "brave_key" in prompt.tool_config:
        tools.append(BraveSearch(api_key=prompt.tool_config["brave_key"]))
    elif prompt.tool == "serp_tool" and "serp_key" in prompt.tool_config:
        tools.append(SerpTool(api_key=prompt.tool_config["serp_key"]))
    elif prompt.tool == "crawl_ai":
        tools.append(CrawlTool())
    elif prompt.tool == "contract_parser":
        # The PDF parser tool expects the file to be already on disk
        file_path = prompt.tool_config.get("contract_parser")
        if not file_path:
            return {"error": "No PDF file uploaded"}
        pdf_tool = PDFKB()
        pdf_tool.file_path = file_path
        tools.append(pdf_tool)

    # 2. Always add the memory tool so the agent can ask for context.
    tools.insert(0, memory_tool)

    # 3. Instruction prompt for the agent
    extra_instructions = """
    - Use the tools you were given to answer the user.
    - You have a persistent memory of past assistant responses; you can ask for them
      via the MemoryTool if needed.
    - Do not ask the user to re‑upload a PDF that has already been parsed.
    """

    # 4. Create the LLM instance
    model = get_model(
        provider=prompt.provider,
        model_id=prompt.id,
        api_key=prompt.api_key
    )

    # 5. Build the agent
    agent = Agent(
        model=model,
        tools=tools,
        tool_choice="auto",
        read_chat_history=True,
        read_tool_call_history=True,
        debug_mode=True,
        markdown=True,
        instructions=extra_instructions
    )

    # 6. Run the conversation
    try:
        response = agent.run(prompt.message)

        # 7. Persist the pair (user + assistant) to the memory DB
        memory_tool.memory.add_message("user", prompt.message)
        memory_tool.memory.add_message("assistant", response.content)

        return {"response": response.content}
    except Exception as exc:
        return {"error": str(exc)}
