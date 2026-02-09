# llm configurations 

LLMConfigs = {
    "openai": {
        "models": [
            {"name": "gpt-5", "value": "gpt-5"},
            {"name": "o3-mini", "value": "o3-mini"},
            {"name": "o1", "value": "o1"},
            {"name": "gpt-4o", "value": "gpt-4o"},
            {"name": "gpt-4o mini", "value": "gpt-4o-mini"},
            {"name": "gpt-4.1", "value": "gpt-4.1"},
            {"name": "gpt-4.1-mini", "value": "gpt-4.1-mini"},
            {"name": "gpt-3.5-turbo", "value": "gpt-3.5-turbo"},
        ]
    },
    "perplexity": {
        "models": [
            {"name": "Sonar with Web Search", "value": "sonar"},
            {"name": "Sonar-pro with Web Search", "value": "sonar-pro"},
        ]
    },
    "gemini": {
        "models": [
            {"name": "Gemini 2.5 Flash-Lite", "value": "gemini-2.5-flash-lite"},
            {"name": "Gemini 2.5 Flash", "value": "gemini-2.5-flash"},
            {"name": "Gemini 2.5 Pro", "value": "gemini-2.5-pro"},
            {"name": "Gemini 2.0 Flash", "value": "gemini-2.0-flash"},
            {"name": "Gemini 2.0 Flash-Lite", "value": "gemini-2.0-flash-lite"},
            {"name": "Gemini 2.0 Pro", "value": "gemini-2.0-pro"},
        ]
    },
    "groq": {
        "models": [
            {"name": "Llama 3.3 70B Versatile", "value": "llama-3.3-70b-versatile"},
            {"name": "Llama 3.1 8B Instant", "value": "llama-3.1-8b-instant"},
            {"name": "Deepseek R1 Distill Llama 70B", "value": "deepseek-r1-distill-llama-70b"},
        ]
    },
    "mistral": {
        "models": [
            {"name": "Codestral Latest", "value": "codestral-latest"},
            {"name": "Mistral Large Latest", "value": "mistral-large-latest"},
            {"name": "Ministral 3B Latest", "value": "ministral-3b-latest"},
            {"name": "Ministral 8B Latest", "value": "ministral-8b-latest"},
            {"name": "Mistral Small Latest", "value": "mistral-small-latest"},
            {"name": "Mistral Embed", "value": "mistral-embed"},
            {"name": "Mistral Moderation Latest", "value": "mistral-moderation-latest"}
        ]
    },
    "ollama":{
        "models":[
            {"name": "Hermes3 8b Llama3.1 q8_0", "value": "hermes3:8b-llama3.1-q8_0"}
        ]
    },
    "claude": {
        "models": [
            {"name": "Claude Haiku 4.5", "value": "claude-3-5-haiku-20241022"},
            {"name": "Claude Sonnet 3.5", "value": "claude-3-5-sonnet-20241022"},
            {"name": "Claude Opus 3", "value": "claude-3-opus-20240229"},
            {"name": "Claude Haiku 3", "value": "claude-3-haiku-20240307"}
        ]
    }

}