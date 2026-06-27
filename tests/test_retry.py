import unittest

from src.services.retry import RetryConfig, run_with_retry


class RetryTests(unittest.TestCase):
    def test_run_with_retry_eventually_succeeds(self):
        calls = {"count": 0}

        def flaky():
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError("temporary")
            return "ok"

        result = run_with_retry(flaky, RetryConfig(attempts=2, backoff_sec=0))

        self.assertTrue(result.ok)
        self.assertEqual(result.value, "ok")
        self.assertEqual(result.attempts, 2)

    def test_run_with_retry_returns_last_error(self):
        result = run_with_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("down")),
            RetryConfig(attempts=2, backoff_sec=0),
        )

        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, RuntimeError)
        self.assertEqual(result.attempts, 2)


if __name__ == "__main__":
    unittest.main()

