import os
import json
import re
from typing import Optional, Dict, Any
import langextract as lx
from langextract.factory import ModelConfig
from langextract import factory
from langchain_community.document_loaders import PyPDFLoader
from agno.tools.toolkit import Toolkit
from dotenv import load_dotenv
from llms.langextract_provider import CustomOPProvider

load_dotenv()

class LangExtractToolkit(Toolkit):
    """Generic LangExtract toolkit."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        super().__init__(name="LangExtract")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_id = model_id
        self.base_url = base_url or os.getenv("BASE_URL")
        self.llm_provider = llm_provider

        self.register(self.extract_from_text)
        self.register(self.extract_from_pdf_with_schema)

    def _build_model(self, schema_json: Dict[str, Any]):
        provider_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "llm_provider": self.llm_provider,
            "response_schema": schema_json,
            "enable_structured_output": True,
            "temperature": 0.0,
        }

        model_config = ModelConfig(
            model_id=self.model_id,
            provider="CustomOPProvider",
            provider_kwargs=provider_kwargs,
        )

        return factory.create_model(model_config)

    def _build_prompt(self, text: str, schema_json: Dict[str, Any]) -> str:
        field_descriptions = {
            k: (v.get("description") if isinstance(v, dict) else "")
            for k, v in schema_json.items()
        }

        return f"""
Extract all fields listed in the schema below.

Rules:
- Use EXACT field names from the schema.
- Output ONLY the extraction JSON.
- No markdown, no explanation, no extra text.

STRICT OUTPUT FORMAT:

<|EXTRACT_START|>
{{
"extractions": [
    {{
    "extraction_class": "FIELD_NAME",
    "extraction_text": "VALUE_FROM_TEXT",
    "attributes": {{}}
    }}
]
}}
<|EXTRACT_END|>

Schema:
{json.dumps(field_descriptions, indent=2)}

Text:
{text}
        """

    def _parse_result(self, result) -> Dict[str, str]:
        clean = {}

        if hasattr(result, "raw_output") and isinstance(result.raw_output, str):
            match = re.search(
                r"<\|EXTRACT_START\|>(.*?)<\|EXTRACT_END\|>",
                result.raw_output,
                re.DOTALL,
            )
            json_text = match.group(1).strip() if match else "{}"

            try:
                parsed = json.loads(json_text)
                for item in parsed.get("extractions", []):
                    clean[item["extraction_class"]] = item.get(
                        "extraction_text", ""
                    )
            except Exception:
                pass
        else:
            try:
                clean = {
                    e.extraction_class: e.extraction_text
                    for e in result.extractions
                }
            except Exception:
                pass

        return clean

    def extract_from_text(self, text: str, schema: Dict[str, Any]) -> str:
        model = self._build_model(schema)
        prompt = self._build_prompt(text, schema)

        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            model=model,
            use_schema_constraints=True,
            fence_output=True,
            debug=False,
        )

        parsed = self._parse_result(result)

        for k in schema.keys():
            parsed.setdefault(k, "")

        return json.dumps(parsed, indent=2, ensure_ascii=False)

    def extract_from_pdf_with_schema(self, pdf_path: str, schema: Dict[str, Any]) -> str:
        """Extract fields from ANY PDF using ANY schema."""
        if not pdf_path or not os.path.exists(pdf_path):
            return json.dumps({k: "" for k in schema.keys()})

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = "\n".join(p.page_content for p in pages)

        return self.extract_from_text(text, schema)

def get_langextract_tool(
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
    llm_provider: Optional[str] = None,
) -> LangExtractToolkit:
    return LangExtractToolkit(
        api_key=api_key,
        model_id=model_id,
        base_url=base_url,
        llm_provider=llm_provider,
    )