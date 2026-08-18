"""HTTP endpoint for authorized hologram analysis."""

from fastapi import APIRouter, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from api.models import HologramResponse
from core.config import COOKIE_SAMESITE, COOKIE_SECURE, HOLOGRAM_SESSION_COOKIE
from core.exceptions import CropSaveError, DocumentSessionAuthorizationError
from services.authorized_hologram_analysis import analyze_authorized_hologram

router = APIRouter(tags=["2. Hologram Analysis"])


@router.post(
    "/v1/hologram/check",
    response_model=HologramResponse,
)

# Cookie'deki karşılaştırma kimliğiyle tek kullanımlık hologram
# servisini çağırır
async def check_hologram(
    request: Request,
    response: Response,
) -> dict[str, object]:
    """Analyze the real front-card crop cached by the latest browser comparison."""
    try:
        document_session_id = request.cookies.get(HOLOGRAM_SESSION_COOKIE, "")
        result = await run_in_threadpool(
            analyze_authorized_hologram, document_session_id
        )
        response.delete_cookie(
            key=HOLOGRAM_SESSION_COOKIE,
            path="/v1/hologram",
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
        )
        return result
    except CropSaveError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except DocumentSessionAuthorizationError as error:
        raise HTTPException(
            status_code=409,
            detail={"message": str(error), "reason_code": error.reason_code},
        ) from error
