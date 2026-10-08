from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import re
import sys
import threading
import traceback
from typing import cast
from types import TracebackType

import structlog
from structlog.typing import EventDict, WrappedLogger
from shared_logging.interfaces import EventLogger, JsonObject, JsonScalar, JsonValue

_context: ContextVar[dict[str, JsonScalar]] = ContextVar("log_context", default={})
_secret_keys = {"authorization", "api_key", "apikey", "access_token", "token", "cookie", "secret",
                "password", "session_id", "gui_session_id", "deck", "hole_cards", "prompt", "raw_response"}
_credentials = re.compile(r"(?i)(authorization|api[_-]?key|(?:access[_-]?)?token|password|secret|cookie|session[_-]?id)"
                          r"([\"']?\s*[:=]\s*[\"']?)([^\s,;\"'}]+)")
_bearer = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_url_query = re.compile(r"(https?://[^\s?]+)\?[^\s]+")
_url_userinfo = re.compile(r"(https?://)[^/\s@]+@")
_route_query = re.compile(r"(/[^\s?]*)\?[^\s]+")
_session_path = re.compile(r"(/api/gui/sessions/)[A-Za-z0-9_-]+")
_event_name = re.compile(r"^[a-z][a-z0-9_.]{0,95}$")
_identifier = re.compile(r"^[a-zA-Z0-9_.-]{1,96}$")
_config: "LoggingConfig | None" = None
_handler: RotatingFileHandler | None = None
_lock = threading.Lock()
_original_sys_hook = sys.excepthook
_original_thread_hook = threading.excepthook
logging.getLogger("shared").addHandler(logging.NullHandler())


def _text(value: str) -> str:
    value = _bearer.sub("Bearer [REDACTED]", value)
    value = _credentials.sub(r"\1\2[REDACTED]", value)
    return _session_path.sub(r"\1[REDACTED]", _route_query.sub(r"\1", _url_userinfo.sub(r"\1", _url_query.sub(r"\1", value))))


def _safe(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, Mapping):
        return {str(k): "[REDACTED]" if str(k).lower() in _secret_keys else _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    return f"[unsupported {type(value).__name__}]"


def fingerprint(value: str) -> str:
    return hashlib.blake2s(value.encode("utf-8"), digest_size=12).hexdigest()


@dataclass(frozen=True)
class LoggingConfig:
    directory: Path
    run_id: str
    process_role: str
    service: str = "holdem"
    level: str = "INFO"
    max_bytes: int = 5_000_000
    backup_count: int = 3

    def __post_init__(self) -> None:
        if not all(_identifier.fullmatch(v) for v in (self.run_id, self.process_role, self.service)):
            raise ValueError("Invalid logging identifier")
        if self.level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError("Invalid log level")
        if self.max_bytes <= 0 or self.backup_count < 1:
            raise ValueError("Invalid rotation limits")

    @classmethod
    def from_environment(cls, role: str, *, service: str = "holdem") -> "LoggingConfig":
        from uuid import uuid4
        return cls(Path(os.environ.get("APP_LOG_DIR", ".data/logs")),
                   os.environ.get("APP_RUN_ID", uuid4().hex), os.environ.get("APP_LOG_ROLE", role), os.environ.get("APP_SERVICE", service),
                   os.environ.get("LOG_LEVEL", "INFO").upper(),
                   int(os.environ.get("LOG_MAX_BYTES", "5000000")), int(os.environ.get("LOG_BACKUP_COUNT", "3")))


def _normalize(logger: WrappedLogger, method_name: str, event: EventDict) -> EventDict:
    config = _config
    assert config is not None
    record = cast(logging.LogRecord, event["_record"])
    business = bool(event.pop("_shared_event", False))
    error: JsonObject | None = None
    info = record.exc_info
    if info is None and isinstance(event.get("exc_info"), tuple):
        info = cast(tuple[type[BaseException], BaseException, TracebackType | None], event["exc_info"])
    if info is not None and info[1] is not None:
        error = {"type": type(info[1]).__name__, "message": _text(str(info[1])),
                 "stack": _text("".join(traceback.format_exception(*info)))}
    context = dict(_context.get())
    context.update(cast(dict[str, JsonScalar], event.get("context", {})) if business else {})
    role = cast(str, event.get("_origin_role", config.process_role)) if business else config.process_role
    normalized: EventDict = {
        "schema_version": 1, "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": "WARNING" if record.levelname == "WARN" else record.levelname,
        "service": config.service, "component": record.name.removeprefix("shared."),
        "process_role": role, "pid": None if role == "browser" else os.getpid(), "run_id": config.run_id,
        "event": str(event.get("event")) if business else "third_party.log",
        "message": _text(str(event.get("message", event.get("event", "")))),
        "context": _safe(context), "data": _safe(event.get("data", {})) if business else {}, "error": error,
    }
    return normalized


structlog.configure(
    processors=[structlog.stdlib.filter_by_level, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    wrapper_class=structlog.stdlib.BoundLogger, logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=False,
)


def configure_logging(config: LoggingConfig) -> Path:
    """Single initialization entry. Each process owns its own rotating file."""
    global _config, _handler
    with _lock:
        root = logging.getLogger()
        path = config.directory.resolve() / config.run_id / f"{config.process_role}-{os.getpid()}.jsonl"
        if _config == config and _handler is not None:
            return path
        if _handler is not None:
            root.removeHandler(_handler)
            _handler.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=config.max_bytes, backupCount=config.backup_count, encoding="utf-8")
        handler.setLevel(config.level)
        handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            processors=[_normalize, structlog.processors.JSONRenderer(ensure_ascii=False, allow_nan=False)],
            keep_exc_info=False,
        ))
        _config, _handler = config, handler
        root.setLevel(config.level)
        root.addHandler(handler)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            external = logging.getLogger(name)
            external.handlers.clear()
            external.propagate = True
            external.setLevel(logging.DEBUG if config.level == "DEBUG" else logging.WARNING if name == "uvicorn.access" else logging.NOTSET)

        def unhandled(kind: type[BaseException], error: BaseException, tb: TracebackType | None) -> None:
            get_logger("process").exception("process.failed", "Unhandled process exception", error=error)
            _original_sys_hook(kind, error, tb)

        def thread_failed(args: threading.ExceptHookArgs) -> None:
            if args.exc_value is not None:
                get_logger("process").exception("process.failed", "Unhandled thread exception", error=args.exc_value)
            _original_thread_hook(args)

        sys.excepthook = unhandled
        threading.excepthook = thread_failed
        return path


def shutdown_logging() -> None:
    global _config, _handler
    with _lock:
        if _handler is not None:
            logging.getLogger().removeHandler(_handler)
            _handler.close()
        _handler, _config = None, None
        sys.excepthook, threading.excepthook = _original_sys_hook, _original_thread_hook


@contextmanager
def process_logging(config: LoggingConfig, *, component: str) -> Iterator[None]:
    """Record startup and failures before releasing the process-owned sink."""
    configure_logging(config)
    log = get_logger(component)
    log.emit("INFO", "process.started", "Process started")
    failed = False
    try:
        yield
    except Exception:
        failed = True
        log.exception("process.failed", "Process failed")
        raise
    finally:
        log.emit("INFO", "process.stopped", "Process stopped", {"failed": failed})
        shutdown_logging()


def logging_environment() -> dict[str, str]:
    config = _config
    if config is None:
        return {}
    return {"APP_RUN_ID": config.run_id, "APP_LOG_DIR": str(config.directory.resolve()),
            "APP_SERVICE": config.service, "LOG_LEVEL": config.level,
            "LOG_MAX_BYTES": str(config.max_bytes), "LOG_BACKUP_COUNT": str(config.backup_count)}


@contextmanager
def context_scope(**values: JsonScalar) -> Iterator[None]:
    token = _context.set({**_context.get(), **values})
    try:
        yield
    finally:
        _context.reset(token)


class StdlibEventLogger(EventLogger):
    def __init__(self, component: str, context: dict[str, JsonScalar] | None = None, origin_role: str | None = None) -> None:
        self.component = component
        self.context = dict(context or {})
        self.origin_role = origin_role

    def bind(self, **context: JsonScalar) -> "EventLogger":
        return StdlibEventLogger(self.component, {**self.context, **context}, self.origin_role)

    def emit(self, level: str, event: str, message: str, data: JsonObject | None = None) -> None:
        self._emit(level, event, message, data, exception=False)

    def exception(self, event: str, message: str, data: JsonObject | None = None, *, error: BaseException | None = None) -> None:
        self._emit("ERROR", event, message, data, exception=True, error=error)

    def _emit(self, level: str, event: str, message: str, data: JsonObject | None, *, exception: bool, error: BaseException | None = None) -> None:
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") or not _event_name.fullmatch(event):
            raise ValueError("Invalid log event or level")
        logger = structlog.stdlib.get_logger(f"shared.{self.component}")
        logger.log(getattr(logging, level), event, message=_text(message), data=_safe(data or {}),
                   context=_safe(self.context), _shared_event=True,
                   _origin_role=self.origin_role or (_config.process_role if _config else "unconfigured"),
                   exc_info=(type(error), error, error.__traceback__) if error is not None else sys.exc_info() if exception else None)


def get_logger(component: str, *, origin_role: str | None = None) -> EventLogger:
    if not _identifier.fullmatch(component):
        raise ValueError("Invalid logger component")
    return StdlibEventLogger(component, origin_role=origin_role)
