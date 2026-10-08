"""Independent second-project logging consumer; only stdlib and shared_logging."""

import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from shared_logging import LoggingConfig, configure_logging, get_logger, shutdown_logging
from shared_logging.observations import ApiCallFailure, ApiResult, ApiUsage, bot_decision, observe_api_call


def operation(url: str) -> ApiResult[str]:
    try:
        with urlopen(Request(url, headers={"Authorization": "Bearer FIXTURE-SECRET"}), timeout=5) as response:
            body = json.load(response)
            usage = body["usage"]
            return ApiResult(str(body["text"]), ApiUsage(int(usage["input_tokens"]), int(usage["output_tokens"])),
                             response.status, str(body["stop_reason"]))
    except HTTPError as error:
        raise ApiCallFailure("Fixture returned HTTP error", http_status=error.code) from error


def main() -> None:
    configure_logging(LoggingConfig.from_environment("other_project", service="example"))
    log = get_logger("other.project")
    log.emit("INFO", "process.started", "Standalone logging consumer started")
    try:
        with bot_decision(log, player_id="fixture-player", hand_id="fixture-hand"):
            result = observe_api_call("local-http-fixture", "fixture-model", lambda: operation(sys.argv[1] + "/success"))
        assert result == "check"
        try:
            with bot_decision(log, player_id="fixture-player", hand_id="fixture-hand"):
                observe_api_call("local-http-fixture", "fixture-model", lambda: operation(sys.argv[1] + "/failure"))
        except ApiCallFailure:
            sys.stderr.write("api_key=FIXTURE-SECRET expected fixture failure\n")
        else:
            raise AssertionError("HTTP failure was not propagated")
        assert "poker" not in sys.modules
        print("Standalone library and real local HTTP fixture verified.")
    finally:
        log.emit("INFO", "process.stopped", "Standalone logging consumer stopped")
        shutdown_logging()


if __name__ == "__main__":
    main()
