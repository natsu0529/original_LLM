from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train  # type: ignore  # noqa: E402


class ComputeLrTests(unittest.TestCase):
    def test_constant_schedule_returns_base(self) -> None:
        for step in (1, 50, 5000):
            self.assertEqual(
                train.compute_lr(
                    step,
                    base_lr=5e-4,
                    min_lr=5e-5,
                    schedule="constant",
                    warmup_steps=0,
                    max_steps=5000,
                ),
                5e-4,
            )

    def test_warmup_ramps_linearly(self) -> None:
        self.assertAlmostEqual(
            train.compute_lr(
                1,
                base_lr=5e-4,
                min_lr=5e-5,
                schedule="cosine",
                warmup_steps=10,
                max_steps=100,
            ),
            5e-5,
        )
        self.assertAlmostEqual(
            train.compute_lr(
                10,
                base_lr=5e-4,
                min_lr=5e-5,
                schedule="cosine",
                warmup_steps=10,
                max_steps=100,
            ),
            5e-4,
        )

    def test_cosine_decays_to_min(self) -> None:
        last = train.compute_lr(
            100,
            base_lr=5e-4,
            min_lr=5e-5,
            schedule="cosine",
            warmup_steps=10,
            max_steps=100,
        )
        self.assertAlmostEqual(last, 5e-5, places=8)

    def test_cosine_midpoint_is_average(self) -> None:
        midpoint = train.compute_lr(
            55,
            base_lr=5e-4,
            min_lr=5e-5,
            schedule="cosine",
            warmup_steps=10,
            max_steps=100,
        )
        # progress = (55 - 10) / 90 = 0.5 → cosine factor = 0.5
        expected = 5e-5 + (5e-4 - 5e-5) * 0.5
        self.assertAlmostEqual(midpoint, expected, places=8)
        # sanity: matches the formula
        self.assertAlmostEqual(
            midpoint,
            5e-5 + (5e-4 - 5e-5) * 0.5 * (1.0 + math.cos(math.pi * 0.5)),
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
