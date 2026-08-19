"""FastAPI application entry point."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.document_check_endpoints import router as document_check_router
from api.hologram_endpoints import router as hologram_router
from core.config import LOG_LEVEL, MODEL_VERSION

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Identity Card Verification Services",
    version=MODEL_VERSION,
    description=(
        "Submit one Base64-encoded identity-card image for side and "
        "real/photocopy risk analysis, or perform an authorized static "
        "hologram check. Image bytes are decoded and analyzed in memory."
    ),
    openapi_tags=[
        {
            "name": "1. Document Analysis",
            "description": (
                "Analyzes one front- or back-side identity-card image."
            ),
        },
        {
            "name": "2. Hologram Analysis",
            "description": "Checks the static hologram ROI on the selected real candidate.",
        },
        {"name": "System", "description": "Application health information."},
    ],
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "none",
        "filter": True,
        "tagsSorter": "alpha",
    },
)

app.include_router(document_check_router)
app.include_router(hologram_router)


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    _request: Request, exception: Exception
) -> JSONResponse:
    """Return a safe response without logging identity image content."""
    logger.exception(
        "Unexpected identity-analysis error: %s",
        type(exception).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred during identity analysis."},
    )


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Return application health and baseline version information."""
    return {"status": "ok", "version": MODEL_VERSION}
