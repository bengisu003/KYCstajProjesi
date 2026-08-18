"""Manage short-lived, one-use document sessions for hologram analysis."""

from dataclasses import dataclass
import secrets
import threading
import time

import numpy as np

from core.config import (
    DOCUMENT_SESSION_TTL_SECONDS,
    MAX_DOCUMENT_SESSIONS,
)
from core.exceptions import DocumentSessionAuthorizationError


@dataclass
class _DocumentSession:
    sample_id: str
    expires_at: float
    card_crop: np.ndarray | None = None
    image_size: dict[str, int] | None = None
    source_frame: str | None = None
    source_filename: str | None = None


_document_sessions: dict[str, _DocumentSession] = {}
_document_sessions_lock = threading.Lock()


def _remove_expired_sessions(now: float) -> None:
    """Remove expired anonymous document sessions while holding the lock."""
    expired_ids = [
        document_session_id
        for document_session_id, session in _document_sessions.items()
        if session.expires_at <= now
    ]
    for document_session_id in expired_ids:
        _document_sessions.pop(document_session_id, None)


def _make_room_for_session() -> None:
    """Evict the earliest-expiring session when the in-memory cache is full."""
    while len(_document_sessions) >= MAX_DOCUMENT_SESSIONS:
        earliest_id = min(
            _document_sessions,
            key=lambda document_session_id: _document_sessions[
                document_session_id
            ].expires_at,
        )
        _document_sessions.pop(earliest_id, None)


def register_real_candidate(
    sample_id: str,
    card_crop: np.ndarray | None = None,
    image_size: dict[str, int] | None = None,
    source_frame: str | None = None,
    source_filename: str | None = None,
) -> str:
    """Authorize and optionally cache one selected real candidate temporarily."""
    document_session_id = secrets.token_urlsafe(24)
    now = time.monotonic()
    cached_crop = (
        np.ascontiguousarray(card_crop).copy()
        if isinstance(card_crop, np.ndarray)
        else None
    )
    with _document_sessions_lock:
        _remove_expired_sessions(now)
        _make_room_for_session()
        _document_sessions[document_session_id] = _DocumentSession(
            sample_id=sample_id,
            expires_at=now + DOCUMENT_SESSION_TTL_SECONDS,
            card_crop=cached_crop,
            image_size=dict(image_size) if image_size is not None else None,
            source_frame=source_frame,
            source_filename=source_filename,
        )
    return document_session_id


def consume_real_candidate(
    document_session_id: str,
) -> tuple[np.ndarray, dict[str, int], str, str | None, str | None]:
    """Atomically consume the cached front real candidate for one hologram check."""
    if not document_session_id or not document_session_id.strip():
        raise DocumentSessionAuthorizationError(
            "An authorized front-document session is required.",
            "DOCUMENT_SESSION_REQUIRED",
        )
    now = time.monotonic()
    with _document_sessions_lock:
        _remove_expired_sessions(now)
        session = _document_sessions.get(document_session_id.strip())
        if session is None or session.card_crop is None:
            raise DocumentSessionAuthorizationError(
                "The document session is missing, expired, or invalid.",
                "DOCUMENT_SESSION_REQUIRED",
            )
        _document_sessions.pop(document_session_id.strip(), None)
    image_size = session.image_size or {
        "width": int(session.card_crop.shape[1]),
        "height": int(session.card_crop.shape[0]),
    }
    return (
        session.card_crop,
        image_size,
        session.sample_id,
        session.source_frame,
        session.source_filename,
    )
