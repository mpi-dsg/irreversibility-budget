import math
import threading
import unittest

from sim.bench import (
    EscrowLedgerTree,
    analytical_model,
    bench_contention,
    bench_latency_vs_depth,
    bench_memory,
    make_path,
    summarize,
)


class EscrowLedgerTreeTests(unittest.TestCase):
    def test_reserve_rejects_invalid_charge(self):
        tree = EscrowLedgerTree()
        path = make_path(tree, 2, budget=100.0)

        for charge in (-1.0, 0.0, math.nan, math.inf):
            with self.subTest(charge=charge):
                with self.assertRaises(ValueError):
                    tree.reserve(path, charge)
                self.assertEqual(path[0].reserved, 0.0)

    def test_reserve_validates_path(self):
        tree = EscrowLedgerTree()
        path = make_path(tree, 2, budget=100.0)
        other = EscrowLedgerTree().new_ledger(100.0)

        bad_paths = ([], [path[0], path[0]], [object()], [other])
        for bad_path in bad_paths:
            with self.subTest(path=bad_path):
                with self.assertRaises(ValueError):
                    tree.reserve(bad_path, 1.0)

    def test_reserve_confirm_cancel_preserve_counters(self):
        tree = EscrowLedgerTree()
        path = make_path(tree, 2, budget=10.0)

        rid = tree.reserve(path, 4.0)
        self.assertIsNotNone(rid)
        self.assertEqual([led.reserved for led in path], [4.0, 4.0])
        self.assertTrue(tree.confirm(rid))
        self.assertFalse(tree.confirm(rid))
        self.assertEqual([led.reserved for led in path], [0.0, 0.0])
        self.assertEqual([led.committed for led in path], [4.0, 4.0])

        rid2 = tree.reserve(path, 3.0)
        self.assertTrue(tree.cancel(rid2))
        self.assertFalse(tree.cancel(rid2))
        self.assertEqual([led.reserved for led in path], [0.0, 0.0])
        self.assertEqual([led.committed for led in path], [4.0, 4.0])

    def test_over_budget_denial_touches_nothing(self):
        tree = EscrowLedgerTree()
        path = make_path(tree, 3, budget=5.0)

        self.assertIsNone(tree.reserve(path, 6.0))
        self.assertEqual([led.reserved for led in path], [0.0, 0.0, 0.0])

    def test_concurrent_reserve_cancel_completes(self):
        tree = EscrowLedgerTree()
        root = tree.new_ledger(1_000_000.0)
        paths = [[root, tree.new_ledger(1_000_000.0)] for _ in range(4)]
        errors = []

        def worker(path):
            try:
                for _ in range(200):
                    rid = tree.reserve(path, 1.0)
                    self.assertIsNotNone(rid)
                    tree.cancel(rid)
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(path,)) for path in paths]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "ledger worker deadlocked")
        self.assertEqual(errors, [])
        self.assertEqual(root.reserved, 0.0)
        self.assertEqual(root.committed, 0.0)


class BenchmarkHelperTests(unittest.TestCase):
    def test_input_validation(self):
        with self.assertRaises(ValueError):
            EscrowLedgerTree().new_ledger(-1)
        with self.assertRaises(ValueError):
            make_path(EscrowLedgerTree(), 0)
        with self.assertRaises(ValueError):
            summarize([])
        with self.assertRaises(ValueError):
            bench_contention([1], depth=1, ops_per_worker=1, warmup_ops=0)
        with self.assertRaises(ValueError):
            bench_memory(depth=1, n_reservations=0)

    def test_small_benchmarks_execute(self):
        latency = bench_latency_vs_depth([1, 2], iters=3, warmup=0)
        self.assertIn("1", latency)

        contention = bench_contention([1], depth=2, ops_per_worker=3,
                                      warmup_ops=0, runs=1)
        self.assertEqual(contention["1"]["total_ops"], 3)

        memory = bench_memory(depth=1, n_reservations=2)
        self.assertEqual(memory["n_reservations"], 2)

        model = analytical_model(latency)
        self.assertIn("fit_reserve_confirm_p50", model)


if __name__ == "__main__":
    unittest.main()
