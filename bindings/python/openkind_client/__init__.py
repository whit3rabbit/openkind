"""Dependency-free, synchronous client for a running openkindd service."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Generic, Literal, NotRequired, TypeVar, TypedDict, Union
from urllib.error import HTTPError
from urllib.request import Request, urlopen


JSONValue = Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]]
State = str | list[JSONValue] | dict[str, JSONValue]
Instructions = str | int | float | bool | list[JSONValue] | dict[str, JSONValue]


class NoulCriteria(TypedDict):
    true: str
    false: str


class NoulQuestion(TypedDict):
    type: Literal["noul"]
    instructions: Instructions
    criteria: NotRequired[NoulCriteria]


class ChoiceQuestion(TypedDict):
    type: Literal["choice"]
    instructions: Instructions
    criteria: dict[str, str | None]


class ScoreQuestion(TypedDict):
    type: Literal["score"]
    instructions: Instructions
    criteria: list[str]


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


class SystemRequest(TypedDict):
    state: State
    model: str
    questions: dict[str, Question]


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(TypedDict):
    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class SystemResponse(TypedDict):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(TypedDict):
    name: str
    description: str
    release_date: str


class ModelsResponse(TypedDict):
    models: list[ModelMetadata]


class Health(TypedDict):
    status: str


T = TypeVar("T")


@dataclass(frozen=True)
class ApiResult(Generic[T]):
    data: T
    request_id: str | None


class ApiError(Exception):
    def __init__(self, status: int, code: str | None, message: str, request_id: str | None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id


class InvalidResponseError(Exception):
    pass


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InvalidResponseError("expected JSON object")
    return value


def _probability(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise InvalidResponseError(f"{label} must be a probability")


def _distribution(value: object, label: str) -> dict[str, object]:
    probabilities = _object(value)
    for key, item in probabilities.items():
        _probability(item, f"{label}.{key}")
    if abs(sum(probabilities.values()) - 1) > 1e-3:
        raise InvalidResponseError(f"{label} must sum to one")
    return probabilities


def validate_response(value: object, request: SystemRequest) -> SystemResponse:
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
            if legend.keys() != probabilities.keys() or any(
                not key.isascii() or not key.isdigit() or len(key) > 10 or int(key) > 0xFFFFFFFF or not isinstance(label, str)
                for key, label in legend.items()
            ):
                raise InvalidResponseError(f"invalid score legend for {ident}")
            score = answer.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= max((int(key) for key in legend), default=0):
                raise InvalidResponseError(f"score out of range for {ident}")
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
        self.base_url = (base_url or os.getenv("OPENKIND_BASE_URL") or os.getenv("TYPESAFE_BASE_URL") or "http://127.0.0.1:18080").rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP URL")
        self.api_key = api_key if api_key is not None else (os.getenv("OPENKIND_API_KEY") or os.getenv("TYPESAFE_API_KEY"))
        self.default_model = default_model or os.getenv("OPENKIND_DEFAULT_MODEL") or os.getenv("TYPESAFE_DEFAULT_MODEL") or "jev-latest"
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def _send(self, path: str, method: str, body: object = None) -> ApiResult[object]:
        headers = {"Accept": "application/json"}
        if self.api_key and path != "/health":
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = None
        if body is not None:
            payload = json.dumps(body, allow_nan=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=payload, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
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
            raise ApiError(error.code, code if isinstance(code, str) else None,
                           message if isinstance(message, str) else f"HTTP {error.code}", request_id) from error

    def evaluate(self, request: SystemRequest) -> ApiResult[SystemResponse]:
        result = self._send("/v1/systemone", "POST", request)
        return ApiResult(validate_response(result.data, request), result.request_id)

    def system_one(self, state: State, questions: dict[str, Question], model: str | None = None) -> ApiResult[SystemResponse]:
        return self.evaluate({"state": state, "model": model or self.default_model, "questions": questions})

    def list_models(self) -> ApiResult[ModelsResponse]:
        result = self._send("/v1/models", "GET")
        models = _object(result.data).get("models")
        if not isinstance(models, list) or any(
            not isinstance(item, dict) or any(not isinstance(item.get(key), str) for key in ("name", "description", "release_date"))
            for item in models
        ):
            raise InvalidResponseError("invalid models response")
        return ApiResult(result.data, result.request_id)  # type: ignore[arg-type]

    def health(self) -> ApiResult[Health]:
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
