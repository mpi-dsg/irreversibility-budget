import json
import tempfile
import unittest
from pathlib import Path

from figures import plot_figs
from figures.plot_figs import wilson_err
from sim.traces import plot as trace_plot


class PlottingTests(unittest.TestCase):
    def test_wilson_err_validates_inputs(self):
        with self.assertRaises(ValueError):
            wilson_err(0.5, 0)
        with self.assertRaises(ValueError):
            wilson_err(1.5, 10)

        lo, hi = wilson_err(0.0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_trace_plot_reads_jsonl_and_writes_png(self):
        results = {
            "cross_agent_taubench": {
                "retail": {
                    "per_class_concentration": {
                        "send": {
                            "concentration_obs": 0.9,
                            "concentration_null": 0.6,
                        }
                    }
                }
            },
            "shared_trigger_agentdojo": {
                "per_suite_attack_success": {
                    "slack": {"rate": 0.1, "successful_injections": 1,
                              "attacked_runs": 10},
                    "banking": {"rate": 0.2, "successful_injections": 2,
                                "attacked_runs": 10},
                    "travel": {"rate": 0.3, "successful_injections": 3,
                               "attacked_runs": 10},
                    "workspace": {"rate": 0.4, "successful_injections": 4,
                                  "attacked_runs": 10},
                }
            },
        }
        tau = [{"n_external": 12, "n_calls": 13, "ext_positions": [12]}]
        dojo = [{"n_external": 56, "n_calls": 57, "ext_positions": [56]}]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plot.png"
            trace_plot.make_plot(results, tau, dojo, out=str(out))
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

            jsonl = Path(tmp) / "rows.jsonl"
            jsonl.write_text(json.dumps(tau[0]) + "\n")
            self.assertEqual(trace_plot.read_jsonl(jsonl), tau)

    def test_trace_plot_load_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results.json").write_text(json.dumps({"ok": True}))
            (root / "parsed_taubench.jsonl").write_text(json.dumps({"a": 1}) + "\n")
            (root / "parsed_agentdojo.jsonl").write_text(json.dumps({"b": 2}) + "\n")

            results, tau, dojo = trace_plot.load_inputs(str(root))

        self.assertEqual(results, {"ok": True})
        self.assertEqual(tau, [{"a": 1}])
        self.assertEqual(dojo, [{"b": 2}])

    def test_paper_figure_functions_write_pdfs(self):
        old_here = plot_figs.HERE
        old_data = plot_figs.data
        old_res = plot_figs.res
        old_r = plot_figs.R

        res = {}
        for key in ("e1_local", "e1_budget", "e1_local_attack", "e1_budget_attack"):
            res[key] = {"exposure": 100.0, "exposure_p95": 120.0, "overdraw": 0.1}
        for bf in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            res[f"e2_bf{bf}"] = {
                "overdraw": 0.1,
                "exec_pre": 0.9,
                "exec_burst": 0.2,
            }
        for eps in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]:
            res[f"e3_eps{eps}"] = {"overdraw": 0.1, "utility_frac": 0.8}

        with tempfile.TemporaryDirectory() as tmp:
            plot_figs.HERE = tmp
            plot_figs.data = {"runs": 10}
            plot_figs.res = res
            plot_figs.R = 100.0
            try:
                plot_figs.fig1()
                plot_figs.fig2()
                plot_figs.fig3()
            finally:
                plot_figs.HERE = old_here
                plot_figs.data = old_data
                plot_figs.res = old_res
                plot_figs.R = old_r

            for name in ("e1_exposure.pdf", "e2_tradeoff.pdf",
                         "e3_mispricing.pdf"):
                self.assertTrue((Path(tmp) / name).exists())


if __name__ == "__main__":
    unittest.main()
