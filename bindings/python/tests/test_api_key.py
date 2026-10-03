import unittest
from unittest import mock

from openkind_client import generate_api_key


class ApiKeyTests(unittest.TestCase):
    def test_generation_uses_32_random_bytes(self):
        with mock.patch("openkind_client.api_key.secrets.token_hex", return_value="ab" * 32) as entropy:
            self.assertEqual(generate_api_key(), "ok_" + "ab" * 32)
        entropy.assert_called_once_with(32)

    def test_generated_keys_have_the_expected_format_and_are_distinct(self):
        keys = {generate_api_key() for _ in range(64)}
        self.assertEqual(len(keys), 64)
        for key in keys:
            self.assertRegex(key, r"^ok_[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
