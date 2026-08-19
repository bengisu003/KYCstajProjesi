"""Shared service-layer exceptions exposed to HTTP endpoint modules."""

# Crop veya JPEG güvenli biçimde yazılamadığında kullanılır.
class CropSaveError(RuntimeError):
    """Raised when an analysis crop cannot be stored safely."""

# Hologram için geçerli bir document session bulunmadığını
# reason_code ile bildirir.
class DocumentSessionAuthorizationError(ValueError):
    """Raised when hologram analysis has no authorized document session."""

    def __init__(self, message: str, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
