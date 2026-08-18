"""Validate and decode bounded image inputs received by API endpoints."""

import base64
import binascii

from fastapi import HTTPException, UploadFile

from core.config import MAX_FRAME_SIZE_MB


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
IMAGE_SIGNATURES = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89PNG\r\n\x1a\n",
}


def decode_base64_image(encoded_image: str, media_type: str) -> bytes:
    """Decode one bounded JPEG/PNG Base64 value without writing it to disk."""
    if media_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG image files are supported.",
        )

    encoded_payload = encoded_image.strip()
    if encoded_payload.lower().startswith("data:"):
        header, separator, encoded_payload = encoded_payload.partition(",")
        expected_header = f"data:{media_type};base64"
        if not separator or header.lower() != expected_header:
            raise HTTPException(
                status_code=422,
                detail="The image data URL does not match media_type.",
            )

    maximum_bytes = MAX_FRAME_SIZE_MB * 1024 * 1024
    maximum_encoded_length = 4 * ((maximum_bytes + 2) // 3)
    if len(encoded_payload) > maximum_encoded_length:
        raise HTTPException(
            status_code=413,
            detail=f"An image can be at most {MAX_FRAME_SIZE_MB} MB.",
        )

    try:
        content = base64.b64decode(encoded_payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail="The image is not valid Base64.",
        ) from error

    if not content:
        raise HTTPException(status_code=422, detail="The decoded image is empty.")
    if len(content) > maximum_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"An image can be at most {MAX_FRAME_SIZE_MB} MB.",
        )
    if not content.startswith(IMAGE_SIGNATURES[media_type]):
        raise HTTPException(
            status_code=422,
            detail="The decoded image content does not match media_type.",
        )
    return content

# JPEG/PNG MIME türü, boş içerik ve maksimum boyutu
# denetleyip byte döndürür
async def read_upload_bytes(upload: UploadFile) -> bytes:
    """Read and validate one bounded image without writing it to disk."""
    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG image files are supported.",
        )

    maximum_bytes = MAX_FRAME_SIZE_MB * 1024 * 1024
    try:
        content = await upload.read(maximum_bytes + 1)
    finally:
        await upload.close()

    if not content:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")
    if len(content) > maximum_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"An image can be at most {MAX_FRAME_SIZE_MB} MB.",
        )
    return content
