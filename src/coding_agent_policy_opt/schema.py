from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class SchemaRegistry:
    """Load and cache the repository's versioned JSON Schema contracts."""

    def __init__(self, schema_dir: Path | None = None) -> None:
        self.schema_dir = schema_dir or Path(__file__).resolve().parents[2] / "schemas"
        self._validators: dict[str, Draft202012Validator] = {}

    def validator(self, name: str) -> Draft202012Validator:
        if name not in self._validators:
            path = self.schema_dir / f"{name}.schema.json"
            schema = json.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            self._validators[name] = Draft202012Validator(schema)
        return self._validators[name]

    def validate(self, name: str, value: Any) -> None:
        self.validator(name).validate(value)

    def schema(self, name: str) -> dict[str, Any]:
        """Return a detached schema value suitable for prompts and audit artifacts."""
        return json.loads(json.dumps(self.validator(name).schema))
