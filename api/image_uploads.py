"""Shared validation and in-memory reading for uploaded images."""

from fastapi import HTTPException, UploadFile

from core.config import MAX_FRAME_SIZE_MB


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}

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
