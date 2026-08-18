"""Endpoint tests for unified single-image document classification."""

import base64
import unittest
from unittest.mock import patch

from fastapi import HTTPException, Response

from api.document_check_endpoints import check_document
from api.models import DocumentCheckRequest, DocumentCheckResponse
from core.config import DOCUMENT_SESSION_TTL_SECONDS, HOLOGRAM_SESSION_COOKIE
from core.exceptions import CropSaveError
from infrastructure.vision.card import InvalidFrameError
from services.document.response_builder import (
    build_document_classification_response,
)


JPEG_BYTES = b"\xff\xd8\xffimage-bytes"


def _payload(
    content: bytes = JPEG_BYTES,
    *,
    filename: str = "card.jpg",
    media_type: str = "image/jpeg",
) -> DocumentCheckRequest:
    return DocumentCheckRequest(
        frame_base64=base64.b64encode(content).decode("ascii"),
        filename=filename,
        media_type=media_type,
    )


def _result(
    *,
    decision: str,
    detected_side: str | None,
    document_session_id: str | None = None,
) -> dict[str, object]:
    return build_document_classification_response(
        analysis={
            "card_detected": True,
            "image_size": {"width": 100, "height": 60},
            "capture_quality": None,
            "features": None,
            "normalized_components": None,
        },
        decision=decision,
        detected_side=detected_side,
        front_side_score=0.9 if detected_side == "front" else 0.2,
        document_score=0.9,
        classification_thresholds={
            "photocopy_max": 0.68,
            "real_min": 0.76,
        },
        reason_codes=["TEST_DECISION"],
        document_session_id=document_session_id,
        source_filename="card.jpg",
    )


class DocumentCheckEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_front_real_result_sets_scoped_http_only_cookie(self) -> None:
        service_result = _result(
            decision="real_candidate",
            detected_side="front",
            document_session_id="document-session-token",
        )
        response = Response()
        with patch(
            "api.document_check_endpoints.classify_document",
            return_value=service_result,
        ) as classify:
            result = await check_document(
                payload=_payload(filename="../uploads/card.jpg"),
                response=response,
            )

        DocumentCheckResponse(**result)
        classify.assert_called_once_with(JPEG_BYTES, "card.jpg")
        set_cookie = response.headers["set-cookie"]
        self.assertIn(
            f"{HOLOGRAM_SESSION_COOKIE}=document-session-token", set_cookie
        )
        self.assertIn(f"Max-Age={DOCUMENT_SESSION_TTL_SECONDS}", set_cookie)
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("Path=/v1/hologram", set_cookie)
        self.assertIn("SameSite=strict", set_cookie)

    async def test_non_authorized_result_deletes_existing_cookie(self) -> None:
        service_result = _result(
            decision="real_candidate",
            detected_side="back",
        )
        response = Response()
        with patch(
            "api.document_check_endpoints.classify_document",
            return_value=service_result,
        ):
            await check_document(payload=_payload(), response=response)

        set_cookie = response.headers["set-cookie"]
        self.assertIn(f'{HOLOGRAM_SESSION_COOKIE}=""', set_cookie)
        self.assertIn("Max-Age=0", set_cookie)
        self.assertIn("Path=/v1/hologram", set_cookie)

    async def test_invalid_base64_is_rejected_before_service_call(self) -> None:
        payload = DocumentCheckRequest(
            frame_base64="not-valid-base64!",
            filename="card.jpg",
            media_type="image/jpeg",
        )
        with (
            patch("api.document_check_endpoints.classify_document") as classify,
            self.assertRaises(HTTPException) as raised,
        ):
            await check_document(payload=payload, response=Response())

        self.assertEqual(raised.exception.status_code, 422)
        classify.assert_not_called()

    async def test_invalid_frame_error_is_mapped_to_422(self) -> None:
        with (
            patch(
                "api.document_check_endpoints.classify_document",
                side_effect=InvalidFrameError("invalid image"),
            ),
            self.assertRaises(HTTPException) as raised,
        ):
            await check_document(payload=_payload(), response=Response())

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail, "invalid image")

    async def test_crop_save_error_is_mapped_to_500(self) -> None:
        with (
            patch(
                "api.document_check_endpoints.classify_document",
                side_effect=CropSaveError("crop failed"),
            ),
            self.assertRaises(HTTPException) as raised,
        ):
            await check_document(payload=_payload(), response=Response())

        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(raised.exception.detail, "crop failed")


if __name__ == "__main__":
    unittest.main()
