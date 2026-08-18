"""Tests for the public FastAPI route and schema contract."""

import unittest

from core.config import MODEL_VERSION
from main import app, health_check


class OpenApiContractTests(unittest.TestCase):
    def test_expected_paths_are_exposed(self) -> None:
        schema = app.openapi()
        self.assertEqual(
            set(schema["paths"]),
            {
                "/health",
                "/v1/document/check",
                "/v1/hologram/check",
            },
        )

    def test_operation_summaries_are_removed(self) -> None:
        for path_item in app.openapi()["paths"].values():
            for operation in path_item.values():
                if isinstance(operation, dict):
                    self.assertNotIn("summary", operation)

    def test_document_check_uses_unified_response_model(self) -> None:
        operation = app.openapi()["paths"]["/v1/document/check"]["post"]
        response_schema = operation["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        self.assertEqual(
            response_schema["$ref"],
            "#/components/schemas/DocumentCheckResponse",
        )

    def test_health_payload_reports_model_version(self) -> None:
        self.assertEqual(
            health_check(),
            {"status": "ok", "version": MODEL_VERSION},
        )


if __name__ == "__main__":
    unittest.main()
