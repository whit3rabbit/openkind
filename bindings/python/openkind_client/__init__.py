"""Dependency-free, synchronous client for a running openkindd service."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Generic, Literal, NotRequired, TypeVar, TypedDict, Union
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


logger = logging.getLogger("openkind_client")
logger.addHandler(logging.NullHandler())

# Mirrors the typesafe-sdk Python SDK's TYPESAFE_LOG_LEVEL handling: the named
# levels are applied once at import and unknown values are ignored.
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "off": logging.CRITICAL + 1,
}


def _apply_log_level_env() -> None:
    value = (os.getenv("OPENKIND_LOG_LEVEL") or os.getenv("TYPESAFE_LOG_LEVEL") or "").strip().lower()
    if value in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS[value])


_apply_log_level_env()


JSONValue = Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]]
State = str | list[JSONValue] | dict[str, JSONValue]
Instructions = str | list[JSONValue] | dict[str, JSONValue]


class NoulCriteria(TypedDict):
    """Text labels defining true and false criteria for binary noul questions."""
    true: str
    false: str


class NoulQuestion(TypedDict):
    """Binary question evaluated to a probability mass between 0 and 1."""
    type: Literal["noul"]
    instructions: Instructions
    criteria: NotRequired[NoulCriteria]


class ChoiceQuestion(TypedDict):
    """Multiple-choice question evaluated across candidate options."""
    type: Literal["choice"]
    instructions: Instructions
    criteria: dict[str, str | None]


class ScoreQuestion(TypedDict):
    """Rubric score question evaluated across defined score levels."""
    type: Literal["score"]
    instructions: Instructions
    criteria: list[str]


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


class SystemRequest(TypedDict):
    """Top-level System One evaluation request containing state, model, and questions."""
    state: State
    model: str
    questions: dict[str, Question]


class NoulAnswer(TypedDict):
    """Evaluated binary decision returning probability mass."""
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(TypedDict):
    """Evaluated multiple-choice decision with selected candidate and probability distribution."""
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(TypedDict):
    """Evaluated rubric decision with expected score, rubric legend, and probabilities."""
    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(TypedDict):
    """Token usage counters reported for input and output."""
    input_tokens: int
    output_tokens: int


class SystemResponse(TypedDict):
    """System One evaluation response containing model name, evaluated answers, and token usage."""
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(TypedDict):
    """Metadata describing an available model profile in the catalog."""
    name: str
    description: str
    release_date: str


class ModelsResponse(TypedDict):
    """Response payload from /v1/models listing available models."""
    models: list[ModelMetadata]


class Health(TypedDict):
    """Daemon health check status response."""
    status: str


T = TypeVar("T")


@dataclass(frozen=True)
class ApiResult(Generic[T]):
    """Generic container wrapping response data and optional request ID header."""
    data: T
    request_id: str | None


class ApiError(Exception):
    """Exception raised when an API request fails with a non-2xx HTTP status code."""

    def __init__(self, status: int, code: str | None, message: str, request_id: str | None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id


class InvalidResponseError(Exception):
    """Exception raised when a response payload violates the wire contract or schema."""
    pass


class _NoRedirectHandler(HTTPRedirectHandler):
    """Keep authenticated requests on their configured origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = build_opener(_NoRedirectHandler)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InvalidResponseError("expected JSON object")
    return value


def _probability(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1 or not math.isfinite(value):
        raise InvalidResponseError(f"{label} must be a probability")


def _distribution(value: object, label: str) -> dict[str, object]:
    probabilities = _object(value)
    for key, item in probabilities.items():
        _probability(item, f"{label}.{key}")
    if abs(sum(probabilities.values()) - 1) > 1e-3:
        raise InvalidResponseError(f"{label} must sum to one")
    return probabilities


def validate_response(value: object, request: SystemRequest) -> SystemResponse:
    """Validate that a decoded response dictionary conforms to the System One wire contract."""
    response = _object(value)
    if not isinstance(response.get("model"), str):
        raise InvalidResponseError("invalid response model")
    answers = _object(response.get("answers"))
    if answers.keys() != request["questions"].keys():
        raise InvalidResponseError("answer IDs do not match question IDs")
    usage = _object(response.get("usage"))
    for field in ("input_tokens", "output_tokens"):
        count = usage.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 0xFFFFFFFF:
            raise InvalidResponseError(f"invalid usage.{field}")
    for ident, question in request["questions"].items():
        answer = _object(answers[ident])
        if answer.get("type") != question["type"]:
            raise InvalidResponseError(f"answer type mismatch for {ident}")
        if question["type"] == "noul":
            _probability(answer.get("noul"), f"{ident}.noul")
        elif question["type"] == "choice":
            choice = answer.get("choice")
            if not isinstance(choice, str) or choice not in question["criteria"]:
                raise InvalidResponseError(f"choice is outside criteria for {ident}")
            probabilities = _distribution(answer.get("probabilities"), f"{ident}.probabilities")
            if probabilities.keys() != question["criteria"].keys():
                raise InvalidResponseError(f"probability keys mismatch for {ident}")
            _probability(answer.get("confidence"), f"{ident}.confidence")
        else:
            legend = _object(answer.get("legend"))
            probabilities = _distribution(answer.get("probabilities"), f"{ident}.probabilities")
            expected_legend = {str(index): label for index, label in enumerate(question["criteria"])}
            if legend != expected_legend or legend.keys() != probabilities.keys():
                raise InvalidResponseError(f"invalid score legend for {ident}")
            score = answer.get("score")
            max_score = max(len(question["criteria"]) - 1, 0)
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= max_score or not math.isfinite(score):
                raise InvalidResponseError(f"score out of range for {ident}")
            expected_score = sum(int(key) * probability for key, probability in probabilities.items())
            if abs(score - expected_score) > 1e-3 * max(max_score, 1):
                raise InvalidResponseError(f"score does not match probabilities for {ident}")
            _probability(answer.get("confidence"), f"{ident}.confidence")
    return response  # type: ignore[return-value]  # Runtime checks above establish the wire shape.


class Client:
    """Synchronous HTTP client. Use the local daemon URL or another compatible System One host."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Initialize the OpenKind HTTP client with base URL, authentication, and timeout settings."""
        self.base_url = (base_url or os.getenv("OPENKIND_BASE_URL") or os.getenv("TYPESAFE_BASE_URL") or "http://127.0.0.1:18080").rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP URL")
        self.api_key = api_key if api_key is not None else (os.getenv("OPENKIND_API_KEY") or os.getenv("TYPESAFE_API_KEY"))
        self.default_model = default_model or os.getenv("OPENKIND_DEFAULT_MODEL") or os.getenv("TYPESAFE_DEFAULT_MODEL") or "jev-latest"
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def _send(self, path: str, method: str, body: object = None) -> ApiResult[object]:
        logger.debug("%s %s", method, path)
        headers = {"Accept": "application/json"}
        if self.api_key and path != "/health":
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = None
        if body is not None:
            payload = json.dumps(body, allow_nan=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=payload, headers=headers, method=method)
        try:
            with _OPENER.open(request, timeout=self.timeout) as response:
                request_id = response.headers.get("x-typesafe-request-id")
                raw = response.read()
                try:
                    return ApiResult(json.loads(raw), request_id)
                except (ValueError, UnicodeDecodeError) as error:
                    raise InvalidResponseError("response is not JSON") from error
        except HTTPError as error:
            request_id = error.headers.get("x-typesafe-request-id")
            try:
                details = json.loads(error.read()).get("error", {})
                code = details.get("code") if isinstance(details, dict) else None
                message = details.get("message") if isinstance(details, dict) else None
            except (ValueError, UnicodeDecodeError, AttributeError):
                code = message = None
            error.close()
            raise ApiError(error.code, code if isinstance(code, str) else None,
                           message if isinstance(message, str) else f"HTTP {error.code}", request_id) from error

    def evaluate(self, request: SystemRequest) -> ApiResult[SystemResponse]:
        """Submit a SystemRequest evaluation payload to /v1/systemone and validate the response."""
        # Keep response checks bound to the submitted rubric when callers reuse mutable dictionaries.
        questions = {ident: question.copy() for ident, question in request["questions"].items()}
        for question in questions.values():
            if "criteria" in question and question["criteria"] is not None:
                question["criteria"] = question["criteria"].copy()
        snapshot = {**request, "questions": questions}
        result = self._send("/v1/systemone", "POST", snapshot)
        return ApiResult(validate_response(result.data, snapshot), result.request_id)

    def system_one(self, state: State, questions: dict[str, Question], model: str | None = None) -> ApiResult[SystemResponse]:
        """Convenience method to evaluate questions against state using the default or specified model."""
        return self.evaluate({"state": state, "model": model or self.default_model, "questions": questions})

    def list_models(self) -> ApiResult[ModelsResponse]:
        """Retrieve the catalog of available models from /v1/models."""
        result = self._send("/v1/models", "GET")
        models = _object(result.data).get("models")
        if not isinstance(models, list) or any(
            not isinstance(item, dict) or any(not isinstance(item.get(key), str) for key in ("name", "description", "release_date"))
            for item in models
        ):
            raise InvalidResponseError("invalid models response")
        return ApiResult(result.data, result.request_id)  # type: ignore[arg-type]

    def health(self) -> ApiResult[Health]:
        """Query the /health endpoint to check daemon availability."""
        result = self._send("/health", "GET")
        if not isinstance(_object(result.data).get("status"), str):
            raise InvalidResponseError("invalid health response")
        return ApiResult(result.data, result.request_id)  # type: ignore[arg-type]


from .server import Server, ServerError


__all__ = [
    "Answer", "ApiError", "ApiResult", "ChoiceAnswer", "ChoiceQuestion", "Client",
    "Health", "Instructions", "InvalidResponseError", "JSONValue", "ModelMetadata", "ModelsResponse",
    "NoulAnswer", "NoulCriteria", "NoulQuestion", "Question", "ScoreAnswer", "ScoreQuestion",
    "Server", "ServerError", "State", "SystemRequest", "SystemResponse", "Usage", "validate_response",
]
