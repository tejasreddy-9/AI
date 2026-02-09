from .openai_provider import OpenAIProvider
from typing import Optional
from configurations import LLMConfigs

class OllamaProvider(OpenAIProvider):
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        base_url = "http://localhost:11434"
        sys_prompt = "You are a helpful assistant."
        self.messages = []

        self.provider_name = "ollama"
        models = [model["value"] for model in LLMConfigs[self.provider_name]["models"]]

        self.models = models
        self.config = {
            "model": "string", 
            "api_key": "string", 
            "temperature": "float", 
            "top_p": "float",
            "max_tokens": "int",
            "system_prompt": "string",
        } 
        
        if "model" in kwargs:
            self.model_name = kwargs.get("model")
        elif "model_name" in kwargs:
            self.model_name = kwargs.get("model_name")
        else:
            self.model_name = models[0]
        
        self.api_key = kwargs.get("api_key", api_key)
        self.temperature = kwargs.get("temperature", 0.1)
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.top_p = kwargs.get("top_p", 0.1)
        self.base_url = kwargs.get("base_url") or base_url
        self.system_prompt = kwargs.get("system_prompt", sys_prompt)