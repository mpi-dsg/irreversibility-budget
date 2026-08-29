#!/usr/bin/env python3
"""Microbenchmark for the hierarchical escrow ledger (irreversibility budget).

Tier-3 "does it scale / what does it cost" evidence for the paper. This is a
*real* implementation of the hierarchical escrow ledger the paper proposes,
not a stub: principals form a tree (agent -> workflow -> tenant), each ledger
holds a budget and separate `reserved` / `committed` counters, and the three
operations are atomic under per-ledger locks:

  reserve(path, charge) -> id
      Acquire every lock on the path (root..leaf) in a fixed GLOBAL order
      (by ledger id) so concurrent reservers can never deadlock. Under the
      locks, check that reserved+committed+charge <= budget on EVERY ledger
      along the path. If all pass, increment `reserved` on each and return an
      id. If any fails, touch nothing and return None (admission denied).
  confirm(id)
      Move charge from `reserved` to `committed` on each ledger on the path.
      Idempotent.
  cancel(id)
      Refund: decrement `reserved` on each ledger on the path. Idempotent.

The shared tenant root is the contention point: every path passes through it,
so its lock serializes the "check budget + bump reserved" critical section
across all principals. This benchmark measures exactly that.

We report hard numbers to sim/bench_results.json and print a summary table:

  1. Per-op latency (ns/op, p50/p99) for reserve+confirm and reserve+cancel,
     single-threaded, vs tree depth d in {1,2,3,4,5}. Expect ~O(d).
  2. Throughput (ops/sec) + p99 latency vs concurrency C in {1,2,4,8,16,32},
     all workers reserving against a SHARED tenant root. Characterizes how
     much the root lock serializes. Reports the Python GIL caveat honestly and
     an analytical model.
  3. Memory per reservation (bytes) and steady-state ledger size vs number of
     live reservations.

Usage: python3 bench.py [--out bench_results.json] [--reps N]
No external installs: stdlib + numpy only.
"""

import argparse
import gc
import json
import math
import os
import statistics
import sys
import threading
import time
import tracemalloc

import numpy as np

# ---------------------------------------------------------------- ledger core


class Ledger:
    """One node in the principal tree. Holds a budget and escrow counters.

    Invariant (maintained under `lock`): reserved >= 0, committed >= 0, and
    reserved + committed <= budget whenever a reservation was admitted.
    """

    __slots__ = ("gid", "budget", "reserved", "committed", "lock", "name")

    def __init__(self, gid, budget, name=""):
        if not isinstance(budget, (int, float)) or not math.isfinite(budget):
            raise ValueError("ledger budget must be a finite number")
        if budget < 0:
            raise ValueError("ledger budget must be non-negative")
        self.gid = gid              # global id -> fixed lock-acquisition order
        self.budget = float(budget)
        self.reserved = 0.0
        self.committed = 0.0
        self.lock = threading.Lock()
        self.name = name


class EscrowLedgerTree:
    """Hierarchical escrow ledger. Principals form a tree; each path from a
    leaf principal up to the root is the sequence of ledgers charged by an
    operation on that principal."""

    def __init__(self):
        self._next_gid = 0
        self.ledgers = {}           # gid -> Ledger
        self._res_lock = threading.Lock()   # guards the reservation table + id
        self._next_res = 0
        self.reservations = {}      # id -> (tuple(path), charge, state)

    def new_ledger(self, budget, name=""):
        gid = self._next_gid
        self._next_gid += 1
        led = Ledger(gid, budget, name)
        self.ledgers[gid] = led
        return led

    @staticmethod
    def _validate_charge(charge):
        if not isinstance(charge, (int, float)) or not math.isfinite(charge):
            raise ValueError("reservation charge must be a finite number")
        if charge <= 0:
            raise ValueError("reservation charge must be positive")
        return float(charge)

    def _validate_path(self, path):
        try:
            ledgers = tuple(path)
        except TypeError as exc:
            raise ValueError("reservation path must be an iterable of ledgers") from exc
        if not ledgers:
            raise ValueError("reservation path must contain at least one ledger")
        seen = set()
        for led in ledgers:
            if not isinstance(led, Ledger):
                raise ValueError("reservation path entries must be Ledger instances")
            if self.ledgers.get(led.gid) is not led:
                raise ValueError("reservation path contains a ledger from another tree")
            if led.gid in seen:
                raise ValueError("reservation path must not contain duplicate ledgers")
            seen.add(led.gid)
        return ledgers

    @staticmethod
    def _ordered(path):
        # Fixed global order (by gid) => no lock-ordering cycle => no deadlock.
        return sorted(path, key=lambda l: l.gid)

    @staticmethod
    def _acquire_all(ordered):
        acquired = []
        try:
            for led in ordered:
                led.lock.acquire()
                acquired.append(led)
            return acquired
        except Exception:
            for led in reversed(acquired):
                led.lock.release()
            raise

    @staticmethod
    def _release_all(acquired):
        for led in reversed(acquired):
            led.lock.release()

    def reserve(self, path, charge):
        """Atomically admit `charge` on every ledger along `path`, or deny.

        Returns a reservation id on success, or None if any ledger on the
        path lacks headroom (in which case no ledger is modified)."""
        path = self._validate_path(path)
        charge = self._validate_charge(charge)
        ordered = self._ordered(path)
        acquired = self._acquire_all(ordered)
        try:
            # All locks held: check headroom on every ledger.
            for led in ordered:
                if led.reserved + led.committed + charge > led.budget:
                    return None     # deny; finally-block releases, no mutation
            for led in ordered:
                led.reserved += charge
        finally:
            self._release_all(acquired)
        # Record the reservation (its own lock; not on the ledger hot path).
        with self._res_lock:
            rid = self._next_res
            self._next_res += 1
            self.reservations[rid] = [tuple(path), charge, "reserved"]
        return rid

    def _begin_reservation_transition(self, rid, pending_state):
        with self._res_lock:
            rec = self.reservations.get(rid)
            if rec is None or rec[2] != "reserved":
                return None
            rec[2] = pending_state
            return rec[0], rec[1]

    def _finish_reservation_transition(self, rid, pending_state, final_state):
        with self._res_lock:
            rec = self.reservations.get(rid)
            if rec is not None and rec[2] == pending_state:
                rec[2] = final_state

    def confirm(self, rid):
        """Move reserved -> committed on each ledger on the path. Idempotent."""
        started = self._begin_reservation_transition(rid, "confirming")
        if started is None:
            return False
        path, charge = started
        ordered = self._ordered(path)
        acquired = self._acquire_all(ordered)
        try:
            for led in ordered:
                led.reserved -= charge
                led.committed += charge
        finally:
            self._release_all(acquired)
        self._finish_reservation_transition(rid, "confirming", "committed")
        return True

    def cancel(self, rid):
        """Refund reserved on each ledger on the path. Idempotent."""
        started = self._begin_reservation_transition(rid, "cancelling")
        if started is None:
            return False
        path, charge = started
        ordered = self._ordered(path)
        acquired = self._acquire_all(ordered)
        try:
            for led in ordered:
                led.reserved -= charge
        finally:
            self._release_all(acquired)
        self._finish_reservation_transition(rid, "cancelling", "cancelled")
        return True


def make_path(tree, depth, budget=1e18):
    """Build a fresh root..leaf path of `depth` ledgers (leaf = index -1)."""
    if not isinstance(depth, int) or depth < 1:
        raise ValueError("path depth must be a positive integer")
    return [tree.new_ledger(budget, name=f"L{i}") for i in range(depth)]


# ---------------------------------------------------------------- timing utils


def pctl(samples_ns, q):
    return float(np.percentile(samples_ns, q))


def summarize(samples_ns):
    a = np.asarray(samples_ns, dtype=np.float64)
    if a.size == 0:
        raise ValueError("cannot summarize an empty sample")
    return {
        "n": int(a.size),
        "mean_ns": float(a.mean()),
        "p50_ns": float(np.percentile(a, 50)),
        "p90_ns": float(np.percentile(a, 90)),
        "p99_ns": float(np.percentile(a, 99)),
        "min_ns": float(a.min()),
        "max_ns": float(a.max()),
    }


# ---------------------------------------------------------------- experiment 1
# Per-op latency vs depth, single-threaded.


def bench_latency_vs_depth(depths, iters, warmup):
    if iters < 1:
        raise ValueError("iters must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    charge = 1.0
    out = {}
    for d in depths:
        # reserve+confirm cycle
        tree = EscrowLedgerTree()
        path = make_path(tree, d)
        rc = np.empty(iters, dtype=np.float64)
        for i in range(warmup):
            rid = tree.reserve(path, charge)
            tree.confirm(rid)
        gc.disable()
        for i in range(iters):
            t0 = time.perf_counter_ns()
            rid = tree.reserve(path, charge)
            tree.confirm(rid)
            rc[i] = time.perf_counter_ns() - t0
        gc.enable()
        # reserve+cancel cycle (fresh tree so budget never fills)
        tree2 = EscrowLedgerTree()
        path2 = make_path(tree2, d)
        rk = np.empty(iters, dtype=np.float64)
        for i in range(warmup):
            rid = tree2.reserve(path2, charge)
            tree2.cancel(rid)
        gc.disable()
        for i in range(iters):
            t0 = time.perf_counter_ns()
            rid = tree2.reserve(path2, charge)
            tree2.cancel(rid)
            rk[i] = time.perf_counter_ns() - t0
        gc.enable()
        out[str(d)] = {
            "reserve_confirm": summarize(rc),
            "reserve_cancel": summarize(rk),
        }
        print(f"  depth={d}: reserve+confirm p50={pctl(rc,50):8.0f}ns "
              f"p99={pctl(rc,99):8.0f}ns | reserve+cancel p50={pctl(rk,50):8.0f}ns "
              f"p99={pctl(rk,99):8.0f}ns")
    return out


# ---------------------------------------------------------------- experiment 2
# Throughput + p99 vs concurrency against a SHARED tenant root.


def _one_contention_run(tree, worker_paths, C, ops_per_worker, warmup_ops):
    if C < 1:
        raise ValueError("concurrency must be positive")
    if ops_per_worker < 1:
        raise ValueError("ops_per_worker must be positive")
    if warmup_ops < 0:
        raise ValueError("warmup_ops must be non-negative")
    latencies = [None] * C
    start_barrier = threading.Barrier(C + 1)

    def worker(wid):
        path = worker_paths[wid]
        lat = np.empty(ops_per_worker, dtype=np.float64)
        for _ in range(warmup_ops):
            rid = tree.reserve(path, 1.0)
            tree.cancel(rid)
        start_barrier.wait()
        for i in range(ops_per_worker):
            t0 = time.perf_counter_ns()
            rid = tree.reserve(path, 1.0)
            tree.cancel(rid)
            lat[i] = time.perf_counter_ns() - t0
        latencies[wid] = lat

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(C)]
    for t in threads:
        t.start()
    start_barrier.wait()
    gc.disable()
    wall0 = time.perf_counter_ns()
    for t in threads:
        t.join()
    wall = time.perf_counter_ns() - wall0
    gc.enable()
    all_lat = np.concatenate(latencies)
    total_ops = C * ops_per_worker
    return total_ops / (wall / 1e9), wall, all_lat


def bench_contention(concurrencies, depth, ops_per_worker, warmup_ops, runs=3):
    """All workers reserve+cancel against paths that share the SAME tenant
    root (and shared upper ancestors), so the root lock is the contention
    point. Budgets are huge so nothing is ever denied -> we measure pure
    lock/critical-section cost, not admission logic. Best-of-`runs` per C to
    reject scheduler/thermal outliers (report the median run)."""
    if depth < 2:
        raise ValueError("depth must be at least 2 for shared-root contention")
    if runs < 1:
        raise ValueError("runs must be positive")
    if any(c < 1 for c in concurrencies):
        raise ValueError("concurrencies must all be positive")
    out = {}
    for C in concurrencies:
        trials = []
        for _ in range(runs):
            tree = EscrowLedgerTree()
            # Shared ancestors: root .. (depth-1 shared levels), then a private
            # leaf per worker. With depth d, ledgers 0..d-2 are shared by all
            # workers; ledger d-1 is the worker's own agent leaf. This is the
            # worst case the paper cares about: every op touches the shared root.
            shared = [tree.new_ledger(1e18, name=f"S{i}") for i in range(depth - 1)]
            worker_paths = []
            for w in range(C):
                leaf = tree.new_ledger(1e18, name=f"leaf{w}")
                worker_paths.append(shared + [leaf])
            trials.append(_one_contention_run(tree, worker_paths, C,
                                              ops_per_worker, warmup_ops))
        trials.sort(key=lambda x: x[0])          # by ops/sec
        ops_per_sec, wall, all_lat = trials[len(trials) // 2]   # median run
        total_ops = C * ops_per_worker
        out[str(C)] = {
            "concurrency": C,
            "total_ops": total_ops,
            "wall_ns": float(wall),
            "ops_per_sec": float(ops_per_sec),
            "ops_per_sec_all_runs": [float(t[0]) for t in trials],
            "latency": summarize(all_lat),
        }
        print(f"  C={C:2d}: {ops_per_sec:12,.0f} ops/s (median of {runs})  "
              f"p50={pctl(all_lat,50):8.0f}ns  p99={pctl(all_lat,99):9.0f}ns")
    return out


# ---------------------------------------------------------------- experiment 3
# Memory per reservation + steady-state ledger size.


def bench_memory(depth, n_reservations):
    if depth < 1:
        raise ValueError("depth must be positive")
    if n_reservations < 1:
        raise ValueError("n_reservations must be positive")
    tree = EscrowLedgerTree()
    path = make_path(tree, depth)

    gc.collect()
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    rids = [tree.reserve(path, 1.0) for _ in range(n_reservations)]
    live = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = live.compare_to(base, "filename")
    total_alloc = sum(s.size_diff for s in stats)
    per_res = total_alloc / n_reservations

    # getsizeof of one reservation record (shallow) for a second estimate
    rec = tree.reservations[rids[0]]
    rec_shallow = (sys.getsizeof(rec) + sys.getsizeof(rec[0])
                   + sys.getsizeof(rec[1]) + sys.getsizeof(rec[2]))

    # Ledger object footprint (fixed per principal, independent of #reservations)
    led = tree.ledgers[0]
    ledger_bytes = sys.getsizeof(led)

    # Steady-state: confirm/cancel do NOT shrink the reservation table (audit
    # trail); measure table growth vs live reservations.
    curve = []
    tree2 = EscrowLedgerTree()
    path2 = make_path(tree2, depth)
    gc.collect()
    tracemalloc.start()
    b2 = tracemalloc.take_snapshot()
    for n in (1000, 5000, 10000, 50000, 100000):
        while len(tree2.reservations) < n:
            tree2.reserve(path2, 1.0)
        snap = tracemalloc.take_snapshot()
        d = sum(s.size_diff for s in snap.compare_to(b2, "filename"))
        curve.append({"n_reservations": n, "table_bytes": int(d),
                      "bytes_per_res": d / n})
    tracemalloc.stop()

    print(f"  per-reservation alloc (tracemalloc): {per_res:.1f} bytes")
    print(f"  reservation record (getsizeof, shallow): {rec_shallow} bytes")
    print(f"  per-principal Ledger object: {ledger_bytes} bytes")
    for c in curve:
        print(f"    {c['n_reservations']:7d} live -> {c['table_bytes']:10,d} B "
              f"({c['bytes_per_res']:.1f} B/res)")

    return {
        "depth": depth,
        "n_reservations": n_reservations,
        "tracemalloc_bytes_per_reservation": float(per_res),
        "getsizeof_record_shallow_bytes": int(rec_shallow),
        "ledger_object_bytes": int(ledger_bytes),
        "steady_state_curve": curve,
    }


# ---------------------------------------------------------------- analytical


def analytical_model(lat_depth):
    """Fit ns/op ~ a + b*d for reserve+confirm to expose the O(d) constant,
    and state the contention model."""
    ds = sorted(int(k) for k in lat_depth)
    xs = np.array(ds, dtype=np.float64)
    ys = np.array([lat_depth[str(d)]["reserve_confirm"]["p50_ns"] for d in ds])
    # least squares y = b*x + a
    b, a = np.polyfit(xs, ys, 1)
    return {
        "fit_reserve_confirm_p50": {"intercept_ns": float(a),
                                    "slope_ns_per_level": float(b)},
        "model": (
            "reserve/confirm/cancel are O(path-depth d): each touches every "
            "ledger on the path once under its lock (a compare + a few float "
            "adds), so serial cost is a + b*d with b ~= per-ledger "
            "lock+arithmetic cost. Under concurrency, only principals whose "
            "paths share an ancestor contend, and they contend ONLY on the "
            "shared ancestors' locks. In the worst case (all paths share the "
            "tenant root) the root critical section is serialized: with a "
            "root critical section of duration s seconds, the ceiling is "
            "1/s ops/s regardless of core count -- Amdahl on the shared root. "
            "Sibling subtrees that share no ancestor scale independently."
        ),
    }


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--reps", type=int, default=200_000,
                    help="single-thread iterations per depth")
    ap.add_argument("--conc-ops", type=int, default=200_000,
                    help="ops per worker in the contention benchmark")
    args = ap.parse_args()

    meta = {
        "python": sys.version,
        "implementation": sys.implementation.name,
        "gil_disabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()
                             is False),
        "numpy": np.__version__,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "timestamp": time.time(),
    }
    print("=" * 74)
    print("Hierarchical escrow ledger microbenchmark")
    print(f"  {sys.implementation.name} {sys.version.split()[0]}  "
          f"GIL {'DISABLED' if meta['gil_disabled'] else 'enabled'}  "
          f"cpus={meta['cpu_count']}")
    print("=" * 74)

    print("\n[1] Per-op latency vs tree depth (single-threaded)")
    lat = bench_latency_vs_depth([1, 2, 3, 4, 5],
                                 iters=args.reps, warmup=5000)

    print("\n[2] Throughput + p99 vs concurrency (SHARED tenant root, depth=3)")
    cont = bench_contention([1, 2, 4, 8, 16, 32], depth=3,
                            ops_per_worker=args.conc_ops, warmup_ops=2000)

    print("\n[3] Memory per reservation + steady-state ledger size (depth=3)")
    mem = bench_memory(depth=3, n_reservations=100_000)

    ana = analytical_model(lat)

    # Headline numbers
    d3 = lat["3"]["reserve_confirm"]
    c1 = cont["1"]["ops_per_sec"]
    c32 = cont["32"]["ops_per_sec"]
    p99_32 = cont["32"]["latency"]["p99_ns"]
    headline = {
        "reserve_confirm_depth3_p50_ns": d3["p50_ns"],
        "reserve_confirm_depth3_p99_ns": d3["p99_ns"],
        "throughput_1_thread_ops_s": c1,
        "throughput_32_thread_ops_s": c32,
        "p99_at_32_threads_ns": p99_32,
        "root_serialization_ratio_32_over_1": c32 / c1,
    }

    results = {
        "meta": meta,
        "headline": headline,
        "exp1_latency_vs_depth": lat,
        "exp2_contention_shared_root": cont,
        "exp3_memory": mem,
        "analytical": ana,
    }

    out_path = args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 74)
    print("HEADLINE")
    print(f"  reserve+confirm @ depth 3: p50={d3['p50_ns']:.0f} ns, "
          f"p99={d3['p99_ns']:.0f} ns")
    print(f"  single-thread throughput : {c1:,.0f} ops/s")
    print(f"  32-thread throughput     : {c32:,.0f} ops/s "
          f"(x{c32/c1:.2f} vs 1 thread)")
    print(f"  p99 latency @ 32 threads : {p99_32:,.0f} ns")
    print(f"  O(d) fit reserve+confirm : {ana['fit_reserve_confirm_p50']['intercept_ns']:.0f}"
          f" + {ana['fit_reserve_confirm_p50']['slope_ns_per_level']:.0f}*d ns")
    print(f"\n  results written to {out_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()
