from .openai_provider import OpenAIProvider
from configurations import LLMConfigs
import requests
import logging
import json
from error_utils import ERROR_MESSAGES
from pydantic import BaseModel

class PerplexityProvider(OpenAIProvider):
    def __init__(self, api_key: str, **kwargs):
        base_url = "https://api.perplexity.ai"
        sys_prompt = "you are a helpful assistant."
        self.messages = []

        self.provider_name = "perplexity"
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
        self.llm_usage = None
        self.isGrounding = False

    def grounded_search(self, payload):
        self.isGrounding = True
        try:
            request_body = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": payload["prompt"]}],
                "stream": False,
                "search_mode": "web",
                "web_search_options": {"search_context_size": "medium"},
                "temperature": self.temperature,      
            }
            
            if self.max_tokens is not None:
                request_body["max_tokens"] = self.max_tokens

            output_format = payload.get("output_format", None)
            if output_format is not None:
                output_format = BaseModel(output_format, "DynamicModel").model_json_schema()
                request_body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"schema": output_format},
                }
                
            req = requests.post(
                url=self.base_url + "/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_body
            )
            
            if req.status_code != 200:
                try:
                    err_json = req.json()
                    msg = err_json.get("error", {}).get("message", req.text)
                except Exception:
                    msg = req.text or "Unknown error"

                if msg == "Message content was empty":
                    msg = "Content of the message is empty"
                else:
                    msg = ERROR_MESSAGES.get(req.status_code, msg)
                err = Exception(msg)
                err.status_code = req.status_code
                raise err
            
            response = req.json()
            message = response["choices"][0]["message"]["content"]
            self.llm_usage = response["usage"]
            citations = response.get("citations", [])
            input_tokens = self.llm_usage["prompt_tokens"]
            output_tokens = self.llm_usage["completion_tokens"]
            total_tokens = self.llm_usage["total_tokens"]
            
            parsed_message = {}
            if message:
                try:
                    parsed_message = json.loads(message)
                except json.JSONDecodeError:
                    parsed_message = {"text": message}
            
            return {
                "status_code": 200,
                "message": parsed_message,
                "citations": citations,
                "metrics": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
            }
        except Exception as e:
            logging.error(f"Search request failed: {e}")
            raise