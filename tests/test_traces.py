import json
import tempfile
import unittest
from pathlib import Path

from sim.traces import analyze
from sim.traces.classify import classify


class TraceClassificationTests(unittest.TestCase):
    def test_classify_known_and_fallback_tools(self):
        self.assertEqual(classify("send_email"), (True, "send", "HIGH"))
        self.assertEqual(classify("get_user"), (False, None, None))
        self.assertEqual(classify("deploy_model"), (True, "other", "MED"))


class TraceAnalysisTests(unittest.TestCase):
    def setUp(self):
        analyze.results.clear()

    def test_read_jsonl_normalizes_attack_success_and_reports_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps({
                "source": "agentdojo",
                "attack_type": "important_instructions",
                "security": False,
            }) + "\n")
            rows = analyze.read_jsonl(path)
            self.assertTrue(rows[0]["attack_success"])

            bad = Path(tmp) / "bad.jsonl"
            bad.write_text("{bad json}\n")
            with self.assertRaisesRegex(ValueError, "bad.jsonl:1"):
                analyze.read_jsonl(bad)

    def test_density_stats_handles_empty_and_zero_call_rows(self):
        empty = analyze.density_stats([], "empty")
        self.assertEqual(empty["n_trajectories"], 0)
        self.assertIsNone(empty["mean_effect_density"])

        zero = analyze.density_stats([{"n_calls": 0, "n_external": 0}], "zero")
        self.assertIsNone(zero["mean_effect_density"])
        self.assertEqual(zero["frac_traj_with_any_effect"], 0.0)

    def test_burstiness_helpers_handle_degenerate_inputs(self):
        self.assertIsNone(analyze.burstiness_B([1]))
        self.assertIsNone(analyze.burstiness_B([0, 0]))
        self.assertIsNone(analyze.fano([0, 0]))
        self.assertEqual(analyze.fano([1, 3]), 0.5)

    def test_shared_trigger_uses_attack_success_not_security_pass(self):
        rows = [
            {
                "attack_type": None,
                "suite": "banking",
                "injection_task": None,
                "security": True,
                "attack_success": False,
                "n_external": 1,
            },
            {
                "attack_type": "important_instructions",
                "suite": "banking",
                "injection_task": "inj",
                "security": False,
                "attack_success": True,
                "n_external": 2,
            },
            {
                "attack_type": "important_instructions",
                "suite": "banking",
                "injection_task": "inj",
                "security": True,
                "attack_success": False,
                "n_external": 0,
            },
        ]

        out = analyze.shared_trigger_agentdojo(rows)

        self.assertEqual(out["benign"]["security_pass_rate"], 1.0)
        self.assertEqual(out["benign"]["attack_success_rate"], 0.0)
        self.assertEqual(out["attacked"]["security_pass_rate"], 0.5)
        self.assertEqual(out["attacked"]["attack_success_rate"], 0.5)
        self.assertEqual(
            out["per_suite_attack_success"]["banking"]["successful_injections"], 1
        )

    def test_cross_agent_empty_and_nonempty_inputs_are_safe(self):
        self.assertEqual(analyze.cross_agent_taubench([]), {})
        rows = []
        for domain in ("retail", "airline"):
            for idx in range(3):
                rows.append({
                    "domain": domain,
                    "task_id": "task",
                    "n_external": 1 if idx else 0,
                    "n_calls": 4,
                    "ext_positions": [idx + 1] if idx else [],
                    "ext_classes": ["send"] if idx else [],
                })
        out = analyze.cross_agent_taubench(rows)
        self.assertEqual(set(out), {"retail", "airline"})

    def test_parse_agentdojo_strict_bad_json(self):
        old_dojo = analyze.DOJO
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "model" / "suite" / "task" / "attack"
            bad.mkdir(parents=True)
            (bad / "injection.json").write_text("{bad json}")
            analyze.DOJO = str(root)
            try:
                with self.assertRaisesRegex(ValueError, "failed to parse"):
                    analyze.parse_agentdojo(strict=True)
                self.assertEqual(analyze.parse_agentdojo(strict=False), [])
            finally:
                analyze.DOJO = old_dojo

    def test_parse_raw_fixture_files(self):
        old_tau, old_dojo = analyze.TAU, analyze.DOJO
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tau = root / "tau"
            tau.mkdir()
            (tau / "gpt-4o-retail.json").write_text(json.dumps([
                {
                    "traj": [
                        {
                            "role": "assistant",
                            "tool_calls": [{"function": {"name": "send_email"}}],
                        }
                    ],
                    "task_id": "task",
                    "trial": 1,
                    "reward": 1,
                }
            ]))

            dojo = root / "dojo"
            run = dojo / "model" / "suite" / "task" / "attack"
            run.mkdir(parents=True)
            (run / "injection.json").write_text(json.dumps({
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "send_email"}}],
                    }
                ],
                "suite_name": "workspace",
                "user_task_id": "user",
                "injection_task_id": "inj",
                "attack_type": "important_instructions",
                "security": False,
                "utility": True,
            }))

            analyze.TAU = str(tau)
            analyze.DOJO = str(dojo)
            try:
                tau_rows = analyze.parse_taubench()
                dojo_rows = analyze.parse_agentdojo()
            finally:
                analyze.TAU = old_tau
                analyze.DOJO = old_dojo

        self.assertEqual(tau_rows[0]["domain"], "retail")
        self.assertTrue(dojo_rows[0]["attack_success"])

    def test_main_with_safe_jsonl_artifacts(self):
        old_here = analyze.HERE
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tau_row = {
                "source": "tau-bench",
                "domain": "retail",
                "task_id": "task",
                "n_calls": 4,
                "n_external": 1,
                "ext_positions": [2],
                "ext_classes": ["send"],
                "ext_sev": ["HIGH"],
            }
            dojo_row = {
                "source": "agentdojo",
                "suite": "workspace",
                "injection_task": "inj",
                "attack_type": "important_instructions",
                "security": False,
                "attack_success": True,
                "utility": True,
                "n_calls": 4,
                "n_external": 1,
                "ext_positions": [2],
                "ext_classes": ["send"],
                "ext_sev": ["HIGH"],
            }
            (root / "parsed_taubench.jsonl").write_text(json.dumps(tau_row) + "\n")
            (root / "parsed_agentdojo.jsonl").write_text(json.dumps(dojo_row) + "\n")
            analyze.HERE = str(root)
            try:
                tau_rows, dojo_rows = analyze.main([])
            finally:
                analyze.HERE = old_here

            results_text = (root / "results.json").read_text()

        self.assertEqual(len(tau_rows), 1)
        self.assertEqual(len(dojo_rows), 1)
        self.assertNotIn("Infinity", results_text)
        json.loads(results_text)


if __name__ == "__main__":
    unittest.main()
