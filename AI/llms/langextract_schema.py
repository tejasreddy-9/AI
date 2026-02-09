from __future__ import annotations
from typing import Any, Sequence
import langextract as lx

class CustomProviderSchema(lx.core.schema.BaseSchema):
    """Custom schema implementation for provider plugin."""

    def __init__(self, schema_dict: dict[str, Any], strict_mode: bool = True):
        self._schema_dict = schema_dict
        self._strict_mode = strict_mode

    @classmethod
    def from_examples(
        cls,
        examples_data: Sequence[lx.data.ExampleData],
        attribute_suffix: str = "_attributes",
    ) -> CustomProviderSchema:
        extraction_classes = set()
        attribute_keys = set()

        for example in examples_data:
            for extraction in example.extractions:
                extraction_classes.add(extraction.extraction_class)
                if extraction.attributes:
                    attribute_keys.update(extraction.attributes.keys())

        schema_dict = {
            "type": "object",
            "properties": {
                "extractions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "extraction_class": {
                                "type": "string",
                                "enum": (
                                    list(extraction_classes)
                                    if extraction_classes
                                    else None
                                ),
                            },
                            "extraction_text": {"type": "string"},
                            "attributes": {
                                "type": "object",
                                "properties": {
                                    key: {"type": "string"}
                                    for key in attribute_keys
                                },
                            },
                        },
                        "required": ["extraction_class", "extraction_text"],
                    },
                },
            },
            "required": ["extractions"],
        }

        if not extraction_classes:
            del schema_dict["properties"]["extractions"]["items"]["properties"]["extraction_class"]["enum"]

        return cls(schema_dict, strict_mode=True)

    def to_provider_config(self) -> dict[str, Any]:
        return {
            "response_schema": self._schema_dict,
            "enable_structured_output": True,
            "output_format": "json",
        }

    @property
    def supports_strict_mode(self) -> bool:
        return self._strict_mode

    @property
    def schema_dict(self) -> dict[str, Any]:
        return self._schema_dict