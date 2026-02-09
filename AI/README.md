# AI Chat Application with FastAPI and Streamlit

This application provides a chat interface with AI models through FastAPI backend and Streamlit frontend, containerized with Docker.

## Features

- Multiple AI provider support (OpenAI, Gemini, Groq, Mistral, Perplexity, Ollama, Claude)
- Tool integrations (Brave Search, SerpAPI, Crawl AI, Contract Parser)
- File upload for contract parsing
- Text-to-speech and speech-to-text capabilities
- Docker containerization for easy deployment

## Prerequisites

- Docker and Docker Compose installed
- API keys for the AI providers you want to use

## Setup

1. Clone this repository
2. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` with your actual API keys

## Running with Docker Compose

```bash
# Build and start the services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

The application will be available at:
- Streamlit UI: http://localhost:8501
- FastAPI backend: http://localhost:8000

## API Endpoints

- `GET /` - Welcome message
- `POST /chat/` - Chat with AI agent
- `POST /upload/` - Upload files

## Usage

1. Open http://localhost:8501 in your browser
2. Select your AI provider and enter the API key
3. Choose a model ID and optional tools
4. Start chatting!

## Development

For local development without Docker:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run FastAPI:
   ```bash
   uvicorn main:app --reload
   ```

3. Run Streamlit (in another terminal):
   ```bash
   streamlit run client_streamlit.py
   ```

## Project Structure

- `main.py` - FastAPI backend
- `client_streamlit.py` - Streamlit frontend
- `llms/` - AI provider implementations
- `tools/` - Tool integrations
- `Dockerfile.fastapi` - FastAPI container
- `Dockerfile.streamlit` - Streamlit container
- `docker-compose.yml` - Multi-service orchestration