from __future__ import annotations

import io
import argparse
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from original_llm.cli import (
    checkpoint_cache_name,
    check_for_updates,
    default_chat_download_url,
    is_newer_version,
    maybe_notify_about_update,
    resolve_retrieval_corpus_dir,
    save_update_check_cache,
)


class VersionComparisonTests(unittest.TestCase):
    def test_is_newer_version_handles_numeric_segments(self) -> None:
        self.assertTrue(is_newer_version("0.1.10", "0.1.2"))
        self.assertFalse(is_newer_version("0.1.0", "0.1.0"))


class DownloadResolutionTests(unittest.TestCase):
    def test_default_chat_download_url_uses_installed_version(self) -> None:
        self.assertEqual(
            default_chat_download_url("0.1.1"),
            "https://github.com/natsu0529/original_LLM/releases/download/v0.1.1/best.pt",
        )

    def test_default_chat_download_url_rejects_non_release_versions(self) -> None:
        self.assertIsNone(default_chat_download_url("0.1.1.dev1"))

    def test_checkpoint_cache_name_is_versioned(self) -> None:
        self.assertEqual(
            checkpoint_cache_name(version_text="0.1.1"),
            "best-v0.1.1.pt",
        )

    def test_checkpoint_cache_name_falls_back_without_release_version(self) -> None:
        self.assertEqual(
            checkpoint_cache_name(version_text="0.1.1.dev1"),
            "best.pt",
        )


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


class RetrievalResolutionTests(unittest.TestCase):
    def test_resolve_retrieval_prefers_chat_checkpoint_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chat_dir = Path(tmp_dir) / "chat_seed_real_persona_v1"
            chat_dir.mkdir()
            args = argparse.Namespace(retrieval_corpus_dir=None)

            resolved = resolve_retrieval_corpus_dir(
                args,
                checkpoint={
                    "args": {
                        "data_dir": str(chat_dir),
                        "reply_loss_label": "相手",
                    }
                },
            )

            self.assertEqual(resolved, str(chat_dir.resolve()))

    def test_resolve_retrieval_prefers_bundled_chat_seed_over_raw_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "txt"
            raw_dir.mkdir()
            args = argparse.Namespace(retrieval_corpus_dir=None)

            resolved = resolve_retrieval_corpus_dir(
                args,
                checkpoint={
                    "args": {
                        "data_dir": str(raw_dir),
                        "reply_loss_label": None,
                    }
                },
            )

            self.assertEqual(
                resolved,
                str((Path(__file__).resolve().parents[1] / "data" / "chat_seed_friend_casual_mix_v1").resolve()),
            )


if __name__ == "__main__":
    unittest.main()
