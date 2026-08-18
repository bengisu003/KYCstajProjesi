"""HTTP endpoint for unified single-image document classification."""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from api.image_uploads import read_upload_bytes
from api.models import DocumentCheckResponse
from core.config import COOKIE_SAMESITE, COOKIE_SECURE, HOLOGRAM_SESSION_COOKIE
from core.exceptions import CropSaveError
from infrastructure.vision.card import InvalidFrameError
from services.document_classification import classify_document


router = APIRouter(tags=["1. Document Analysis"])


@router.post(
    "/v1/document/check",
    response_model=DocumentCheckResponse,
)
async def check_document(
    response: Response,
    frame: UploadFile = File(
        ...,
        description="Front- or back-side identity-card image.",
    ),
) -> dict[str, object]:
    """Detect the card side, then estimate real/photocopy evidence."""
    try:
        source_filename = Path(frame.filename or "frame").name
        frame_bytes = await read_upload_bytes(frame)
        result = await run_in_threadpool(
            classify_document,
            frame_bytes,
            source_filename,
        )

        document_session_id = result.get("document_session_id")
        hologram_ready = (
            result.get("hologram_ready") is True
            and isinstance(document_session_id, str)
            and bool(document_session_id)
        )
        if hologram_ready:
            response.set_cookie(
                key=HOLOGRAM_SESSION_COOKIE,
                value=document_session_id,
                max_age=int(result["hologram_session_expires_in_seconds"]),
                httponly=True,
                samesite=COOKIE_SAMESITE,
                secure=COOKIE_SECURE,
                path="/v1/hologram",
            )
        else:
            response.delete_cookie(
                key=HOLOGRAM_SESSION_COOKIE,
                path="/v1/hologram",
                secure=COOKIE_SECURE,
                samesite=COOKIE_SAMESITE,
            )
        return result
    except InvalidFrameError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except CropSaveError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
