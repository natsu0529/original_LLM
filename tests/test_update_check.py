from __future__ import annotations

import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from original_llm.cli import (
    check_for_updates,
    is_newer_version,
    maybe_notify_about_update,
    save_update_check_cache,
)


class VersionComparisonTests(unittest.TestCase):
    def test_is_newer_version_handles_numeric_segments(self) -> None:
        self.assertTrue(is_newer_version("0.1.10", "0.1.2"))
        self.assertFalse(is_newer_version("0.1.0", "0.1.0"))


class UpdateCheckTests(unittest.TestCase):
    def test_check_for_updates_uses_fresh_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "update-check.json"
            save_update_check_cache(
                {
                    "checked_at": time.time(),
                    "latest_version": "0.2.0",
                },
                cache_path,
            )

            result = check_for_updates(
                "0.1.0",
                cache_path=cache_path,
                fetch_latest_version=lambda: self.fail("fetch should not run"),
            )

            self.assertEqual(result.latest_version, "0.2.0")
            self.assertTrue(result.update_available)

    def test_maybe_notify_about_update_prints_once_per_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "update-check.json"
            first_stream = io.StringIO()
            second_stream = io.StringIO()

            with mock.patch("original_llm.cli.package_version_string", return_value="0.1.0"):
                maybe_notify_about_update(
                    stream=first_stream,
                    cache_path=cache_path,
                    fetch_latest_version=lambda: "0.1.1",
                )
                maybe_notify_about_update(
                    stream=second_stream,
                    cache_path=cache_path,
                    fetch_latest_version=lambda: "0.1.1",
                )

            self.assertIn("A newer version of original-llm is available: 0.1.1", first_stream.getvalue())
            self.assertIn("uv tool upgrade original-llm", first_stream.getvalue())
            self.assertEqual(second_stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
