from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Iterator, Sequence

import langextract as lx
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput
import langextract.resolver as resolver

from .langextract_schema import CustomProviderSchema

def patched_extract_and_parse_content(self, input_string: str) -> dict[str, Any]:
    """
    Patched version of _extract_and_parse_content that tolerates newlines
    and spaces before/after markers.
    """
    pattern = r"(?s)<\|EXTRACT_START\|>\s*(\{[\s\S]*?\})\s*<\|EXTRACT_END\|>"
    match = re.search(pattern, input_string)

    if not match:
        match = re.search(r"\{[\s\S]*\}", input_string)
        if not match:
            raise ValueError("No valid extraction markers or JSON found.")

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}

resolver.Resolver._extract_and_parse_content = patched_extract_and_parse_content

@lx.providers.registry.register(r"^opai")
@dataclasses.dataclass(init=False)
class CustomOPProvider(BaseLanguageModel):
    """LangExtract provider adapter that routes requests to internal LLM providers via ProviderFactory."""
    model_id: str
    api_key: str | None
    temperature: float
    response_schema: dict[str, Any] | None = None
    enable_structured_output: bool = False

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not api_key:
            raise lx.exceptions.InferenceConfigError(
                "API key required. Pass api_key parameter."
            )

        self.raw_model_id = model_id
        explicit_provider = kwargs.get("llm_provider")
        parsed_provider: str | None = None
        parsed_model_id: str = model_id

        if "/" in model_id:
            prefix, rest = model_id.split("/", 1)
            parsed_provider = prefix.lower()
            parsed_model_id = rest

        self.llm_provider = (
            explicit_provider
            or parsed_provider
            or "openai"
        ).lower()

        self.model_id = parsed_model_id
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature

        self.response_schema = kwargs.get("response_schema")
        self.enable_structured_output = kwargs.get(
            "enable_structured_output", False
        )

        self._provider = None

    @classmethod
    def get_schema_class(cls) -> type[lx.schema.BaseSchema] | None:
        return CustomProviderSchema

    def apply_schema(self, schema_instance: lx.schema.BaseSchema | None) -> None:
        super().apply_schema(schema_instance)
        if schema_instance:
            config = schema_instance.to_provider_config()
            self.response_schema = config.get("response_schema")
            self.enable_structured_output = config.get(
                "enable_structured_output", False
            )
        else:
            self.response_schema = None
            self.enable_structured_output = False

    def chat(self, prompt: str, system_prompt: str = None):
        """Simple chat method for LangExtract integration."""
        from .provider_factory import ProviderFactory
        
        if self._provider is None:
            factory = ProviderFactory()
            self._provider = factory.get_provider_instance(
                provider_name=self.llm_provider,
                api_key=self.api_key,
                model=self.model_id,
                base_url=self.base_url,
                temperature=self.temperature
            )
        
        if system_prompt:
            self._provider.setSystemPrompt(system_prompt)
        
        return self._provider.chatCompletion(prompt)

    def infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs: Any,
    ) -> Iterator[Sequence[ScoredOutput]]:

        provider_name = (
            kwargs.get("provider_name")
            or self.llm_provider
            or "openai"
        )

        system_prompt = "You are a strict json field extraction model"

        if self._provider is None:
            from .provider_factory import ProviderFactory
            factory = ProviderFactory()
            self._provider = factory.get_provider_instance(
                provider_name=provider_name,
                api_key=self.api_key,
                model=self.model_id,
                base_url=self.base_url,
                temperature=self.temperature
            )

        provider = self._provider

        for prompt in batch_prompts:
            try:
                raw_output = self.chat(
                    prompt=prompt,
                    system_prompt=system_prompt,
                )

                raw_output = (
                    raw_output.replace("```json", "")
                              .replace("```", "")
                              .strip()
                )

                match = re.search(
                    r"<\|EXTRACT_START\|>(.*?)<\|EXTRACT_END\|>",
                    raw_output,
                    re.DOTALL,
                )

                json_text = match.group(1).strip() if match else "{}"

                try:
                    json_data = json.loads(json_text)
                except json.JSONDecodeError:
                    json_data = {"extractions": []}

                extractions = json_data.get("extractions", [])

                if not isinstance(extractions, list):
                    extractions = [
                        {
                            "extraction_class": k,
                            "extraction_text": v,
                            "attributes": {},
                        }
                        for k, v in json_data.items()
                    ]

                final_output = (
                    "<|EXTRACT_START|>\n"
                    + json.dumps(
                        {"extractions": extractions},
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n<|EXTRACT_END|>"
                )

                yield [ScoredOutput(output=final_output, score=1.0)]

            except Exception as e:
                raise lx.exceptions.InferenceRuntimeError(
                    f"LLM error: {str(e)}",
                    original=e,
                ) from e