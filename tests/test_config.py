"""Tests for optional environment overrides and fail-fast validation."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from core import config


class ConfigTests(unittest.TestCase):
    def test_safe_defaults_work_without_environment_variables(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config._positive_int_from_env("TEST_INT", 10), 10)
            self.assertFalse(config._boolean_from_env("TEST_BOOL", False))
            self.assertEqual(
                config._choice_from_env("TEST_CHOICE", "strict", {"strict", "lax"}),
                "strict",
            )

    def test_environment_values_are_parsed_case_insensitively(self) -> None:
        with patch.dict(
            os.environ,
            {"TEST_BOOL": "TRUE", "TEST_CHOICE": "warning"},
            clear=True,
        ):
            self.assertTrue(config._boolean_from_env("TEST_BOOL", False))
            self.assertEqual(
                config._choice_from_env(
                    "TEST_CHOICE", "INFO", {"INFO", "WARNING"}
                ),
                "WARNING",
            )

    def test_invalid_operational_values_fail_fast(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "0"}, clear=True):
            with self.assertRaises(RuntimeError):
                config._positive_int_from_env("TEST_INT", 10)
        with patch.dict(os.environ, {"TEST_BOOL": "sometimes"}, clear=True):
            with self.assertRaises(RuntimeError):
                config._boolean_from_env("TEST_BOOL", False)

    def test_crop_directory_cannot_escape_project_root(self) -> None:
        with patch.dict(os.environ, {"TEST_CROP_DIR": ".."}, clear=True):
            with self.assertRaises(RuntimeError):
                config._project_subdirectory_from_env(
                    "TEST_CROP_DIR", "cropped_images"
                )

    def test_exported_crop_directory_is_below_project_root(self) -> None:
        self.assertIsInstance(config.CROP_OUTPUT_DIR, Path)
        self.assertTrue(config.CROP_OUTPUT_DIR.is_relative_to(config.PROJECT_ROOT))
        self.assertNotEqual(config.CROP_OUTPUT_DIR, config.PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
