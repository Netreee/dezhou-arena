from poker.domain.types import ErrorCode


class RuleViolation(ValueError):
    """A rejected command. No table snapshot should be saved for it."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
