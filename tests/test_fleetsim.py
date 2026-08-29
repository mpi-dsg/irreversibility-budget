import math
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

from sim import fleetsim


class FleetSimTests(unittest.TestCase):
    def test_unknown_regime_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown admission regime"):
            fleetsim.simulate(1, "budegt")

    def test_attack_requires_enough_agents_for_colluders(self):
        with self.assertRaisesRegex(ValueError, "frag_agents"):
            fleetsim.simulate(1, "budget", attack="asap", n_agents=5)

    def test_zero_burst_multiplier_is_honored(self):
        default = fleetsim.simulate(10, "budget")
        no_burst = fleetsim.simulate(10, "budget", p_burst_mult=0.0)

        self.assertLess(no_burst["n_denied"], default["n_denied"])
        self.assertLess(default["exec_burst"], no_burst["exec_burst"])

    def test_invalid_numeric_inputs_raise(self):
        bad_kwargs = [
            {"windows": 0},
            {"budget_frac": -0.1},
            {"gamma": 1.0},
            {"ref_recovery": 0.0},
            {"p_burst_mult": 100.0},
            {"breaker_delay": -1},
        ]
        for kwargs in bad_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    fleetsim.simulate(1, "budget", **kwargs)

    def test_mc_single_run_has_zero_ci_not_nan(self):
        out = fleetsim.mc(1, 100, regime="budget", p_burst_mult=0.0)

        ci_values = [value for key, value in out.items() if key.endswith("_ci")]
        self.assertTrue(ci_values)
        self.assertTrue(all(value == 0.0 for value in ci_values))
        self.assertFalse(any(math.isnan(value) for value in out.values()))

    def test_main_run_limit_matches_seed_spacing(self):
        self.assertEqual(fleetsim.MAX_RUNS_WITH_UNIQUE_SWEEP_SEEDS, 500)

    def test_main_smoke_run_writes_outputs(self):
        old_argv = sys.argv
        with tempfile.TemporaryDirectory() as tmp:
            sys.argv = ["fleetsim.py", "--runs", "1", "--out", tmp]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    fleetsim.main()
            finally:
                sys.argv = old_argv

            self.assertTrue((Path(tmp) / "results.json").exists())
            self.assertTrue((Path(tmp) / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
