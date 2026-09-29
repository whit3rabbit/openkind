import logging
import os
import unittest
from unittest import mock

import openkind_client


class LogEnvTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("openkind_client")
        self.original_level = self.logger.level

    def tearDown(self):
        self.logger.setLevel(self.original_level)

    def test_typesafe_log_level_is_applied(self):
        with mock.patch.dict(os.environ, {"TYPESAFE_LOG_LEVEL": " debug "}):
            openkind_client._apply_log_level_env()
        self.assertEqual(self.logger.level, logging.DEBUG)

    def test_openkind_log_level_wins(self):
        with mock.patch.dict(os.environ, {"OPENKIND_LOG_LEVEL": "warn", "TYPESAFE_LOG_LEVEL": "error"}):
            openkind_client._apply_log_level_env()
        self.assertEqual(self.logger.level, logging.WARNING)

    def test_unknown_value_is_ignored(self):
        self.logger.setLevel(logging.INFO)
        with mock.patch.dict(os.environ, {"TYPESAFE_LOG_LEVEL": "chatty"}):
            openkind_client._apply_log_level_env()
        self.assertEqual(self.logger.level, logging.INFO)

    def test_off_disables_critical_records(self):
        with mock.patch.dict(os.environ, {"TYPESAFE_LOG_LEVEL": "off"}):
            openkind_client._apply_log_level_env()
        self.assertFalse(self.logger.isEnabledFor(logging.CRITICAL))


if __name__ == "__main__":
    unittest.main()
