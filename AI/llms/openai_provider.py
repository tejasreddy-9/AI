from openai import OpenAI, AsyncOpenAI
from configurations import LLMConfigs
from typing import Any, List
import logging
import json
from pydantic import BaseModel
from error_utils import ERROR_MESSAGES
from .base_provider import BaseProvider

class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        base_url = "https://api.openai.com/v1"
        sys_prompt = "You are a helpful assistant."
        self.messages = []
        self.save_messages = False
        self.isGrounding = False
        self.provider_name = "openai"
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
        self.reasoning_effort = kwargs.get("reasoning_effort", None)

    def getModels(self) -> List[str]:
        return self.models
    
    def getProviderName(self) -> str:
        return self.provider_name
    
    def chatCompletion(self, prompt: str, save_messages: bool = False) -> str:
        messages = self.getMessages(prompt)
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_tokens,
            messages=messages
        )

        if response.choices[0].message is not None:
            response_text = response.choices[0].message.content
        else:
            response_text = "" 

        if save_messages:
            self.messages.append({"role": "assistant", "content": response_text})
        else:
            self.messages = []

        return response_text
    
    async def asyncChatCompletion(self, prompt: str, save_messages: bool = False) -> Any:
        messages = self.getMessages(prompt)
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        response = await client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            messages=messages
        )

        if response.choices[0].message is not None:
            response_text = response.choices[0].message.content
        else:
            response_text = ""

        if save_messages:
            self.messages.append({"role": "assistant", "content": response_text})
        else:
            self.messages = []

        return response_text

    def grounded_search(self, payload):
        self.isGrounding = True
        try:
            request_body = {
                "tools": [{"type": "web_search"}],
                "model": self.model_name,
                "input": [
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": payload["prompt"]
                    },
                ],
            }
            if not (self.model_name.startswith("gpt-5") or self.model_name.startswith("gpt-5.1")):
                request_body["temperature"] = self.temperature
                if self.max_tokens is not None:
                    request_body["max_output_tokens"] = self.max_tokens
            
            output_format = payload.get("output_format", None)
            if output_format is not None:
                output_format = BaseModel(output_format, "DynamicModel")
                request_body["text_format"] = output_format

            client = OpenAI(api_key=self.api_key)
            response = client.responses.parse(**request_body)
            
            message = None
            citations = []
            for item in response.output:
                if item.type == "message":
                    message = item.content[0].text
                    if hasattr(item.content[0], "annotations"):
                        annotations = item.content[0].annotations
                        citations = [a.url for a in annotations if hasattr(a, "url")]
                    break
            
            self.llm_usage = response.usage
            input_tokens = self.llm_usage.input_tokens
            output_tokens = self.llm_usage.output_tokens
            total_tokens = self.llm_usage.total_tokens
            self.llm_usage = self.llm_usage.to_dict()
            
            parsed_message = {}
            if message:
                try:
                    parsed_message = json.loads(message)
                except json.JSONDecodeError:
                    parsed_message = {"text": message}
            
            return {
                "status_code": 200,
                "message": parsed_message,
                "metrics": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                },
                "citations": citations
            }
        except Exception as e:
            logging.error(f"Search request failed: {e}")
            status_code = getattr(e, "status_code", 500)
            body_msg = None
            if hasattr(e, "body") and isinstance(e.body, dict):
                body_msg = e.body.get("message")
            message = body_msg or str(e)
            message = ERROR_MESSAGES.get(status_code, message)
            err = Exception(message)
            err.status_code = status_code
            raise err