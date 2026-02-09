from error_utils import ERROR_MESSAGES
from .openai_provider import OpenAIProvider
from configurations import LLMConfigs
import requests
import logging
import json
from pydantic import BaseModel

class GeminiProvider(OpenAIProvider):
    def __init__(self, api_key: str, **kwargs):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        sys_prompt = "You are a helpful assistant."
        self.messages = []

        self.provider_name = "gemini"
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

    def grounded_search(self, payload):
        self.isGrounding = True
        try:
            prompt_text = payload.get("prompt", "").strip()
            if not prompt_text: 
                prompt_text = " "
            
            request_body = {
                "contents": [
                    {
                        "parts": [{"text": prompt_text}],
                    }
                ],
                "tools": [
                    {"google_search": {}}
                ]
            }
            
            if prompt_text and prompt_text != " ":
                gen_cfg = {"temperature": self.temperature}
                if self.max_tokens is not None:
                    gen_cfg["max_output_tokens"] = self.max_tokens
                request_body["generationConfig"] = gen_cfg
            
            output_format = payload.get("output_format", None)
            if output_format:
                output_format_str = str(output_format).lower()
                has_date_fields = any(keyword in output_format_str for keyword in ["{{date}}"])
                    
                base_instruction = " Return structured outputs in JSON format only (no extra text). which follows all json rules, no extra text. Return 'N/A' for missing data or when information is not found. Return only what is asked without any additional text. For integer,double,long return a valid number, else 'N/A'. For boolean return true/false, else 'N/A'."
                date_rule = ""
                if has_date_fields:
                    date_rule = " Dates: ISO format YYYY-MM-DDTHH:mm:ss.SSSZ (e.g. '2023-01-01T00:00:00.000Z'), if date is not found return 'N/A'."
                    
                request_body["contents"][0]["parts"][0]['text'] += f"\n\nFormat: {output_format}\nRules: {base_instruction}{date_rule}"
                
            req = requests.post(
                url=self.base_url,
                headers={"x-goog-api-key": self.api_key},
                json=request_body
            )
            
            try:
                response = req.json()
            except ValueError:
                logging.error(f"Failed to parse response JSON: {req.text}")
                raise

            if "error" in response and isinstance(response["error"], dict):
                msg = response["error"].get("message", "Unknown error occurred")
                code = response["error"].get("code", 500)
                if "api key not valid" in msg.lower():
                    code = 401
                msg = ERROR_MESSAGES.get(code, msg)
                err = Exception(msg)
                err.status_code = code
                raise err

            message = response["candidates"][0]["content"]["parts"][0]["text"]
            citations = []
            if "groundingChunks" in response["candidates"][0]["groundingMetadata"]:
                data = response["candidates"][0]["groundingMetadata"]["groundingChunks"]
                citations = [item["web"]["uri"] for item in data if "web" in item and "uri" in item["web"]]

            self.llm_usage = response.get("usageMetadata", {})
            input_tokens = self.llm_usage.get("promptTokenCount", 0)
            output_tokens = self.llm_usage.get("candidatesTokenCount", 0) + self.llm_usage.get("toolUseCompletionTokenCount", 0) + self.llm_usage.get("thoughtsTokenCount", 0)
            total_tokens = self.llm_usage.get("totalTokenCount", 0)

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