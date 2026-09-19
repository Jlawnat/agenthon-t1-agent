from __future__ import annotations

import unittest

from agent.candidate_prompt import (
    build_precision_guidance,
)


class PrecisionGuidanceTests(unittest.TestCase):
    def test_delta_hedging_guidance(self) -> None:
        notes = build_precision_guidance(
            "Discrete delta hedging with transaction costs, "
            "cash dividends, and time-varying implied volatility."
        )
        joined = " ".join(notes).lower()

        self.assertIn("previous close", joined)
        self.assertIn("terminal liquidation", joined)

    def test_event_study_guidance(self) -> None:
        notes = build_precision_guidance(
            "Earnings event study with market model abnormal returns."
        )
        joined = " ".join(notes).lower()

        self.assertIn("trading-day", joined)
        self.assertIn("prediction-error", joined)

    def test_compound_poisson_fft_guidance(self) -> None:
        notes = build_precision_guidance(
            "Aggregate loss for a compound Poisson model via FFT."
        )
        joined = " ".join(notes).lower()

        self.assertIn("exp(lambda*(phi_x-1))", joined)
        self.assertIn("wrap-around", joined)

    def test_kirk_margrabe_guidance(self) -> None:
        notes = build_precision_guidance(
            "Price Kirk spread options and Margrabe exchange options."
        )
        joined = " ".join(notes).lower()

        self.assertIn("k=0", joined)
        self.assertIn("f1=s1", joined)

    def test_unrelated_task_does_not_receive_special_notes(self) -> None:
        notes = build_precision_guidance(
            "Read a CSV and calculate the arithmetic mean."
        )

        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
