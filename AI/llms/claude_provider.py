from anthropic import Anthropic, AsyncAnthropic
from configurations import LLMConfigs
from typing import Any, List
import logging
from error_utils import ERROR_MESSAGES
from .base_provider import BaseProvider

class ClaudeProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        base_url = "https://api.anthropic.com"
        sys_prompt = "You are a helpful assistant."
        self.messages = []
        self.save_messages = False
        self.isGrounding = False
        self.provider_name = "claude"
        models = [model["value"] for model in LLMConfigs[self.provider_name]["models"]]

        self.models = models
        self.config = {
            "model": "string",
            "api_key": "string",
            "temperature": "float",
            "top_p": "float",
            "max_tokens": "int",
            "system_prompt": "string"
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

    def getModels(self) -> List[str]:
        return self.models

    def getProviderName(self) -> str:
        return self.provider_name

    def chatCompletion(self, prompt: str, save_messages: bool = False) -> str:
        messages = self.getMessages(prompt)
        client = Anthropic(api_key=self.api_key, base_url=self.base_url)

        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})

        response = client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            system=system_message,
            messages=anthropic_messages
        )

        response_text = response.content[0].text if response.content else ""

        if save_messages:
            self.messages.append({"role": "assistant", "content": response_text})
        else:
            self.messages = []

        return response_text

    async def asyncChatCompletion(self, prompt: str, save_messages: bool = False) -> Any:
        messages = self.getMessages(prompt)
        client = AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)

        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})

        response = await client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            system=system_message,
            messages=anthropic_messages
        )

        response_text = response.content[0].text if response.content else ""

        if save_messages:
            self.messages.append({"role": "assistant", "content": response_text})
        else:
            self.messages = []

        return response_text

    def grounded_search(self, payload):
        self.isGrounding = True
        try:
            # Claude doesn't have built-in web search like OpenAI
            # This would need to be implemented with external tools
            raise NotImplementedError("Grounded search not implemented for Claude")
        except Exception as e:
            logging.error(f"Search request failed: {e}")
            status_code = getattr(e, "status_code", 500)
            message = ERROR_MESSAGES.get(status_code, str(e))
            err = Exception(message)
            err.status_code = status_code
            raise err