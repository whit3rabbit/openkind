import copy
import unittest
from unittest import mock

from openkind_client import ApiResult, Client, InvalidResponseError, validate_response


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.request = {
            "state": "ticket", "model": "mock", "questions": {
                "severity": {"type": "score", "instructions": "Rate severity",
                             "criteria": ["low", "medium", "high"]},
            },
        }
        self.response = {
            "model": "mock", "answers": {
                "severity": {
                    "type": "score", "score": 1.5,
                    "legend": {"0": "low", "1": "medium", "2": "high"},
                    "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                    "confidence": 0.6,
                },
            }, "usage": {"input_tokens": 2, "output_tokens": 1},
        }

    def test_accepts_expected_score_and_scaled_rounding(self):
        validate_response(self.response, self.request)
        self.response["answers"]["severity"]["score"] += 0.0015
        validate_response(self.response, self.request)

    def test_rejects_changed_score_rubric(self):
        answer = self.response["answers"]["severity"]
        for legend in (
            {"0": "high", "1": "medium", "2": "low"},
            {"0": "low", "1": "medium"},
            {"0": "low", "1": "medium", "2": "high", "3": "critical"},
            {"00": "low", "1": "medium", "2": "high"},
        ):
            with self.subTest(legend=legend):
                answer["legend"] = legend
                answer["probabilities"] = {key: float(index == 0) for index, key in enumerate(legend)}
                answer["score"] = 0
                with self.assertRaises(InvalidResponseError):
                    validate_response(self.response, self.request)

    def test_rejects_score_inconsistent_with_distribution(self):
        self.response["answers"]["severity"]["score"] = 0
        with self.assertRaises(InvalidResponseError):
            validate_response(self.response, self.request)

    def test_rejects_invalid_numeric_values_without_overflow(self):
        for field in ("score", "confidence"):
            for value in (True, float("nan"), float("inf"), 10 ** 400):
                with self.subTest(field=field, value=value):
                    response = copy.deepcopy(self.response)
                    response["answers"]["severity"][field] = value
                    with self.assertRaises(InvalidResponseError):
                        validate_response(response, self.request)
        self.response["answers"]["severity"]["probabilities"]["1"] = 10 ** 400
        with self.assertRaises(InvalidResponseError):
            validate_response(self.response, self.request)

    def test_evaluate_checks_submitted_request_snapshot(self):
        client = Client(api_key="")

        def send(_path, _method, body):
            self.assertIs(body["state"], self.request["state"])
            self.assertIs(body["questions"]["severity"]["instructions"],
                          self.request["questions"]["severity"]["instructions"])
            self.assertEqual(body["questions"]["severity"]["criteria"], ["low", "medium", "high"])
            # Simulate another thread reusing the caller's question dictionary during HTTP I/O.
            self.request["questions"]["severity"]["criteria"][:] = ["high", "medium", "low"]
            return ApiResult(self.response, "request-123")

        with mock.patch.object(client, "_send", side_effect=send):
            result = client.evaluate(self.request)
        self.assertEqual(result.request_id, "request-123")
        self.assertEqual(result.data["answers"]["severity"]["legend"]["0"], "low")


if __name__ == "__main__":
    unittest.main()
