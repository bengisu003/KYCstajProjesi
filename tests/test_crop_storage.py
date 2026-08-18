"""Tests for classified front/back crop storage routing."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from core.config import CROP_OUTPUT_DIR
from infrastructure.storage import crop_storage


class CropStorageTests(unittest.TestCase):
    def test_general_storage_routes_back_photocopy_crop(self) -> None:
        card_crop = np.zeros((4, 6, 3), dtype=np.uint8)
        with (
            patch.object(crop_storage, "cleanup_expired_crop_files"),
            patch.object(crop_storage, "write_jpeg") as write_jpeg,
        ):
            result = crop_storage.save_document_card_crop(
                card_crop,
                "photocopy",
                "back",
            )

        target_path = write_jpeg.call_args.args[0]
        self.assertEqual(target_path.parent, CROP_OUTPUT_DIR / "photocopy" / "back")
        self.assertEqual(result["sample_id"], target_path.stem)
        self.assertEqual(
            result["card_crop"],
            str(Path("cropped_images") / "photocopy" / "back" / target_path.name),
        )

    def test_invalid_label_and_side_are_rejected_before_writing(self) -> None:
        card_crop = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch.object(crop_storage, "write_jpeg") as write_jpeg:
            with self.assertRaises(ValueError):
                crop_storage.save_document_card_crop(
                    card_crop,
                    "unknown",  # type: ignore[arg-type]
                    "front",
                )
            with self.assertRaises(ValueError):
                crop_storage.save_document_card_crop(
                    card_crop,
                    "real",
                    "unknown",  # type: ignore[arg-type]
                )
        write_jpeg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
