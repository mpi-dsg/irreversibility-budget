#!/usr/bin/env python3
"""
Test the paper's premise on REAL agent traces:
  Do fleets of LLM agents produce CORRELATED bursts of external (irreversible)
  effects, such that many individually-permitted actions create large aggregate
  irreversible exposure?

Datasets (real, public tool-call trajectories):
  * tau-bench historical_trajectories: 4 files, retail+airline customer-service
    agents (gpt-4o, claude-3.5-sonnet). Each trial = one agent solving one task;
    many trials share a task_id  -> lets us test cross-agent correlation given a
    shared trigger.
  * AgentDojo runs: banking/workspace/travel/slack agents across ~30 models,
    with prompt-injection attacks. A single injected instruction is a literal
    SHARED trigger encountered by many heterogeneous agents -> the sharpest
    real-data analog of the paper's correlated latent driver.

Outputs (in this directory):
  parsed_taubench.jsonl, parsed_agentdojo.jsonl  -- one line per trajectory
  results.json                                   -- all metrics
  burstiness.png                                 -- figure

Fresh parsing needs cloned raw datasets. Set either:
  IRREVERSIBILITY_TRACE_RAW_ROOT=/path/with/tau-bench/and/agentdojo
or:
  IRREVERSIBILITY_TAUBENCH_DIR=/path/to/historical_trajectories
  IRREVERSIBILITY_AGENTDOJO_DIR=/path/to/runs
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import classify

random.seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))

RAW_ROOT = os.environ.get(
    "IRREVERSIBILITY_TRACE_RAW_ROOT",
    os.path.join(HERE, "raw"),
)
TAU = os.environ.get(
    "IRREVERSIBILITY_TAUBENCH_DIR",
    os.path.join(RAW_ROOT, "tau-bench", "historical_trajectories"),
)
DOJO = os.environ.get(
    "IRREVERSIBILITY_AGENTDOJO_DIR",
    os.path.join(RAW_ROOT, "agentdojo", "runs"),
)

results = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tool_calls_of(msg):
    out = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function")
        name = fn.get("name") if isinstance(fn, dict) else fn
        if name:
            out.append(name)
    return out

def seq_from_messages(messages):
    """Ordered list of tool-call names across a trajectory."""
    seq = []
    for m in messages:
        if m.get("role") == "assistant" or m.get("tool_calls"):
            seq.extend(tool_calls_of(m))
    return seq

def burstiness_B(intervals):
    """Goh-Barabasi burstiness of inter-event step intervals.
    B in [-1,1]: -1 periodic, 0 Poisson-random, +1 maximally bursty."""
    if len(intervals) < 2:
        return None
    mu = st.mean(intervals)
    sd = st.pstdev(intervals)
    if mu + sd == 0:
        return None
    return (sd - mu) / (sd + mu)

def fano(counts):
    """Index of dispersion (Fano factor) of per-window counts. 1=Poisson."""
    if len(counts) < 2:
        return None
    mu = st.mean(counts)
    if mu == 0:
        return None
    return st.pvariance(counts) / mu

def read_jsonl(path):
    rows = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL in {path}:{lineno}: {exc}") from exc
                if row.get("source") == "agentdojo" and "attack_success" not in row:
                    attack_type = row.get("attack_type")
                    row["attack_success"] = (
                        not bool(row.get("security"))
                        if attack_type not in (None, "none") else False
                    )
                rows.append(row)
    return rows

def require_raw_inputs():
    missing = [p for p in (TAU, DOJO) if not os.path.isdir(p)]
    if missing:
        raise SystemExit(
            "Raw trace inputs are missing. Use the committed parsed JSONL artifacts "
            "for normal artifact use, or set IRREVERSIBILITY_TRACE_RAW_ROOT, "
            "IRREVERSIBILITY_TAUBENCH_DIR, and/or IRREVERSIBILITY_AGENTDOJO_DIR "
            "before running with --fresh. Missing: " + ", ".join(missing)
        )

# ---------------------------------------------------------------------------
# PARSE  tau-bench
# ---------------------------------------------------------------------------
def parse_taubench():
    rows = []
    for f in sorted(glob.glob(os.path.join(TAU, "*.json"))):
        tag = os.path.basename(f)[:-5]
        model = "sonnet-35" if "sonnet" in tag else "gpt-4o"
        domain = "airline" if "airline" in tag else "retail"
        with open(f) as fh:
            d = json.load(fh)
        for t in d:
            seq = seq_from_messages(t["traj"])
            flags = [classify(n) for n in seq]
            ext_idx = [i for i, fl in enumerate(flags) if fl[0]]
            rows.append({
                "source": "tau-bench",
                "file": tag, "model": model, "domain": domain,
                "task_id": t["task_id"], "trial": t.get("trial", 0),
                "reward": t.get("reward"),
                "n_calls": len(seq),
                "n_external": len(ext_idx),
                "ext_positions": ext_idx,
                "ext_classes": [flags[i][1] for i in ext_idx],
                "ext_sev": [flags[i][2] for i in ext_idx],
                "seq": seq,
            })
    return rows

# ---------------------------------------------------------------------------
# PARSE  AgentDojo  (subset: full corpus is 36k files; parse all)
# ---------------------------------------------------------------------------
def parse_agentdojo(strict=True):
    rows = []
    skipped = []
    files = glob.glob(os.path.join(DOJO, "**", "*.json"), recursive=True)
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append((f, exc))
            continue
        msgs = d.get("messages")
        if not msgs:
            continue
        # path: runs/<model>/<suite>/<user_task>/<attack_type>/<injection>.json
        rel = os.path.relpath(f, DOJO).split(os.sep)
        model = rel[0]
        seq = seq_from_messages(msgs)
        flags = [classify(n) for n in seq]
        ext_idx = [i for i, fl in enumerate(flags) if fl[0]]
        attack_type = d.get("attack_type")
        security_passed = bool(d.get("security"))
        rows.append({
            "source": "agentdojo",
            "model": model,
            "suite": d.get("suite_name"),
            "user_task": d.get("user_task_id"),
            "injection_task": d.get("injection_task_id"),
            "attack_type": attack_type,
            "security": security_passed,  # True = security property held
            "attack_success": (not security_passed)
                              if attack_type not in (None, "none") else False,
            "utility": bool(d.get("utility")),
            "n_calls": len(seq),
            "n_external": len(ext_idx),
            "ext_positions": ext_idx,
            "ext_classes": [flags[i][1] for i in ext_idx],
            "ext_sev": [flags[i][2] for i in ext_idx],
            "seq": seq,
        })
    if skipped and strict:
        examples = "; ".join(f"{path}: {exc}" for path, exc in skipped[:5])
        raise ValueError(
            f"failed to parse {len(skipped)} AgentDojo JSON files; "
            f"examples: {examples}"
        )
    return rows

# ---------------------------------------------------------------------------
# METRIC (i): external-effect density
# ---------------------------------------------------------------------------
def density_stats(rows, label):
    if not rows:
        d = {
            "n_trajectories": 0,
            "mean_calls_per_traj": None,
            "mean_external_per_traj": None,
            "median_external_per_traj": None,
            "max_external_in_a_traj": None,
            "mean_effect_density": None,
            "frac_traj_with_any_effect": None,
        }
        results.setdefault("density", {})[label] = d
        return d
    dens = [r["n_external"] / r["n_calls"] for r in rows if r["n_calls"] > 0]
    ncalls = [r["n_calls"] for r in rows]
    next_ = [r["n_external"] for r in rows]
    frac_any = sum(1 for r in rows if r["n_external"] > 0) / len(rows)
    d = {
        "n_trajectories": len(rows),
        "mean_calls_per_traj": round(st.mean(ncalls), 3),
        "mean_external_per_traj": round(st.mean(next_), 3),
        "median_external_per_traj": st.median(next_),
        "max_external_in_a_traj": max(next_),
        "mean_effect_density": round(st.mean(dens), 4) if dens else None,
        "frac_traj_with_any_effect": round(frac_any, 4),
    }
    results.setdefault("density", {})[label] = d
    return d

# ---------------------------------------------------------------------------
# METRIC (ii): within-trajectory burstiness
# ---------------------------------------------------------------------------
def burstiness_stats(rows, label, nwin=5):
    """Do a single agent's external effects cluster in step-position rather than
    spread uniformly?  Pool inter-event intervals -> Goh-Barabasi B; and per-traj
    Fano factor over nwin equal windows.  Compared against Poisson (B~0, Fano~1)."""
    Bs, fanos = [], []
    all_rel = []  # relative positions of effects (for temporal-cluster view)
    for r in rows:
        pos = r["ext_positions"]
        n = r["n_calls"]
        if r["n_external"] >= 2 and n >= 4:
            intervals = [pos[0]] + [pos[i+1]-pos[i] for i in range(len(pos)-1)]
            b = burstiness_B(intervals)
            if b is not None:
                Bs.append(b)
            # Fano over windows
            edges = [int(round(k*n/nwin)) for k in range(nwin+1)]
            counts = []
            for k in range(nwin):
                counts.append(sum(1 for p in pos if edges[k] <= p < edges[k+1]))
            fv = fano(counts)
            if fv is not None:
                fanos.append(fv)
        if n > 1:
            all_rel.extend([p/(n-1) for p in pos])
    d = {
        "n_traj_used": len(Bs),
        "mean_burstiness_B": round(st.mean(Bs), 4) if Bs else None,
        "median_burstiness_B": round(st.median(Bs), 4) if Bs else None,
        "frac_bursty_B_gt_0": round(sum(1 for b in Bs if b > 0)/len(Bs), 4) if Bs else None,
        "mean_fano_5win": round(st.mean(fanos), 4) if fanos else None,
        "mean_rel_position_of_effects": round(st.mean(all_rel), 4) if all_rel else None,
        "note": "B: -1 periodic, 0 Poisson, +1 bursty. Fano>1 => over-dispersed vs Poisson.",
    }
    results.setdefault("within_traj_burstiness", {})[label] = d
    return d

# ---------------------------------------------------------------------------
# METRIC (iii-a): cross-agent correlation given a SHARED TASK (tau-bench)
# ---------------------------------------------------------------------------
def cross_agent_taubench(rows):
    """Group trials by (domain, task_id). Trials of the same task = a fleet of
    agents hitting the SAME trigger. If effects were idiosyncratic per agent, the
    per-task fraction emitting an effect ~ marginal rate q (concentration ~0.5 in
    the balanced case). If the shared trigger DRIVES effects, that fraction is
    pushed toward 0 or 1 -> high concentration. We compare observed concentration
    to a permutation null that breaks the task<->effect link but preserves the
    marginal effect rate."""
    out = {}
    for domain in ("retail", "airline"):
        drows = [r for r in rows if r["domain"] == domain]
        if not drows:
            continue
        by_task = collections.defaultdict(list)
        for r in drows:
            by_task[r["task_id"]].append(r)
        # keep tasks with >=3 trials
        tasks = {k: v for k, v in by_task.items() if len(v) >= 3}
        if not tasks:
            continue
        # ---- "emits any external effect" indicator ----
        def conc_for(indicator):
            vals = []
            for k, v in tasks.items():
                p = st.mean([indicator(r) for r in v])
                vals.append(max(p, 1 - p))
            return st.mean(vals)
        ind_any = lambda r: 1.0 if r["n_external"] > 0 else 0.0
        C_obs = conc_for(ind_any)
        # marginal
        q = st.mean([ind_any(r) for r in drows])
        # permutation null: shuffle indicator labels across all trials
        labels = [ind_any(r) for r in drows]
        sizes = [len(v) for v in tasks.values()]
        null = []
        for _ in range(500):
            random.shuffle(labels)
            idx = 0; vals = []
            for s in sizes:
                grp = labels[idx:idx+s]; idx += s
                p = st.mean(grp); vals.append(max(p, 1-p))
            null.append(st.mean(vals))
        mu_n, sd_n = st.mean(null), st.pstdev(null)
        z = (C_obs - mu_n) / sd_n if sd_n > 0 else None

        # ---- count consistency: CV of external-effect COUNT within a task ----
        cvs = []
        for k, v in tasks.items():
            cnts = [r["n_external"] for r in v]
            m = st.mean(cnts)
            if m > 0:
                cvs.append(st.pstdev(cnts)/m)
        # ---- per effect-CLASS concentration, w/ null + bimodality ----
        # For each irreversible-effect class: is WHICH effect fires determined by
        # the task (shared trigger) or by the agent? Compare observed
        # concentration to a permutation null; measure bimodality = fraction of
        # tasks where the fleet is near-unanimous (|p-0.5|>0.4).
        classes = ("pay", "refund", "delete", "create", "update", "send",
                   "post")
        class_conc = {}
        tsizes = [len(v) for v in tasks.values()]
        for c in classes:
            indc = lambda r, c=c: 1.0 if c in r["ext_classes"] else 0.0
            marg = st.mean([indc(r) for r in drows])
            if marg == 0:
                continue
            C = conc_for(indc)
            # bimodality
            ps = []
            for k, v in tasks.items():
                ps.append(st.mean([indc(r) for r in v]))
            bimod = st.mean([1.0 if abs(p-0.5) > 0.4 else 0.0 for p in ps])
            # permutation null
            labs = [indc(r) for r in drows]
            nl = []
            for _ in range(300):
                random.shuffle(labs)
                idx = 0; vv = []
                for s in tsizes:
                    g = labs[idx:idx+s]; idx += s
                    p = st.mean(g); vv.append(max(p, 1-p))
                nl.append(st.mean(vv))
            mn, sn = st.mean(nl), st.pstdev(nl)
            class_conc[c] = {
                "marginal_rate": round(marg, 4),
                "concentration_obs": round(C, 4),
                "concentration_null": round(mn, 4),
                "z_vs_null": round((C-mn)/sn, 2) if sn > 0 else None,
                "frac_tasks_fleet_near_unanimous": round(bimod, 4),
            }
        # ---- temporal alignment: rel position of first effect, per-task CV ----
        pos_cvs = []
        for k, v in tasks.items():
            firsts = []
            for r in v:
                if r["ext_positions"] and r["n_calls"] > 1:
                    firsts.append(r["ext_positions"][0]/(r["n_calls"]-1))
            if len(firsts) >= 3 and st.mean(firsts) > 0:
                pos_cvs.append(st.pstdev(firsts)/st.mean(firsts))

        out[domain] = {
            "n_tasks_ge3trials": len(tasks),
            "n_trials": len(drows),
            "marginal_rate_any_effect_q": round(q, 4),
            "concentration_observed": round(C_obs, 4),
            "concentration_null_mean": round(mu_n, 4),
            "concentration_null_sd": round(sd_n, 4),
            "z_score_vs_null": round(z, 2) if z is not None else None,
            "mean_within_task_count_CV": round(st.mean(cvs), 4) if cvs else None,
            "per_class_concentration": class_conc,
            "mean_within_task_firsteffect_position_CV":
                round(st.mean(pos_cvs), 4) if pos_cvs else None,
            "interpretation": ("concentration>>null and low count-CV => a shared "
                               "trigger drives correlated effects across the fleet"),
        }
    results["cross_agent_taubench"] = out
    return out

# ---------------------------------------------------------------------------
# METRIC (iii-b): SHARED-TRIGGER correlated burst (AgentDojo injections)
# ---------------------------------------------------------------------------
def shared_trigger_agentdojo(rows):
    """A prompt injection is a single instruction planted in a shared resource
    (webpage/email/message). Many heterogeneous agents (different user_tasks,
    different models) encounter the SAME injection. If it succeeds, they all
    perform the SAME injected external effect -> a correlated irreversible burst
    from one latent driver. We measure the fleet-wide success rate per injection
    trigger and the external-effect lift of attacked vs benign runs."""
    attacked = [r for r in rows if r["attack_type"] == "important_instructions"]
    benign   = [r for r in rows if r["attack_type"] in (None, "none")]

    def eff_rate(rs):
        if not rs:
            return {
                "n": 0,
                "mean_external_per_traj": None,
                "frac_with_any_effect": None,
                "security_pass_rate": None,
                "attack_success_rate": None,
            }
        n = [r["n_external"] for r in rs]
        return {
            "n": len(rs),
            "mean_external_per_traj": round(st.mean(n), 4),
            "frac_with_any_effect": round(sum(1 for x in n if x > 0)/len(rs), 4),
            "security_pass_rate": round(st.mean([r["security"] for r in rs]), 4),
            "attack_success_rate": round(st.mean([r.get("attack_success", False)
                                                  for r in rs]), 4),
        }

    # Per shared trigger = (suite, injection_task): across all user_tasks & models,
    # what fraction of the fleet fires the injected effect (security==True)?
    by_trig = collections.defaultdict(list)
    for r in attacked:
        by_trig[(r["suite"], r["injection_task"])].append(r)
    trig_rates = []
    for k, v in by_trig.items():
        if len(v) >= 20:
            trig_rates.append(st.mean([r.get("attack_success", False) for r in v]))
    # concentration of the shared-trigger success (how aligned the fleet is)
    conc_trig = st.mean([max(p, 1-p) for p in trig_rates]) if trig_rates else None

    # value/exposure proxy: security-success runs by suite (banking=money etc.)
    by_suite = collections.defaultdict(lambda: [0, 0])
    for r in attacked:
        by_suite[r["suite"]][0] += 1
        by_suite[r["suite"]][1] += int(r.get("attack_success", False))
    suite_succ = {k: {"attacked_runs": v[0], "successful_injections": v[1],
                      "rate": round(v[1]/v[0], 4)} for k, v in by_suite.items()}

    benign_rate = eff_rate(benign)
    attacked_rate = eff_rate(attacked)
    benign_mean = benign_rate["mean_external_per_traj"]
    attacked_mean = attacked_rate["mean_external_per_traj"]
    lift = None
    if attacked_mean is not None and benign_mean is not None:
        lift = round(attacked_mean / max(benign_mean, 1e-9), 3)

    out = {
        "benign": benign_rate,
        "attacked": attacked_rate,
        "external_effect_lift_attacked_vs_benign": lift,
        "n_shared_triggers_ge20runs": len(trig_rates),
        "mean_per_trigger_success_rate": round(st.mean(trig_rates), 4) if trig_rates else None,
        "max_per_trigger_success_rate": round(max(trig_rates), 4) if trig_rates else None,
        "fleet_alignment_concentration": round(conc_trig, 4) if conc_trig else None,
        "per_suite_attack_success": suite_succ,
        "interpretation": ("one planted instruction drives the same irreversible "
                           "effect across many heterogeneous agents; per-trigger "
                           "attack-success rate = degree of correlated fleet response"),
    }
    results["shared_trigger_agentdojo"] = out
    return out

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="parse raw cloned datasets instead of committed JSONL")
    ap.add_argument("--skip-invalid-json", action="store_true",
                    help="skip malformed AgentDojo raw JSON files during --fresh")
    ap.add_argument("--use-pickle-cache", action="store_true",
                    help="explicitly load _parsed_cache.pkl if JSONL is absent")
    ap.add_argument("--write-pickle-cache", action="store_true",
                    help="write _parsed_cache.pkl after a fresh parse")
    args = ap.parse_args(argv)

    results.clear()
    cache = os.path.join(HERE, "_parsed_cache.pkl")
    parsed_tau = os.path.join(HERE, "parsed_taubench.jsonl")
    parsed_dojo = os.path.join(HERE, "parsed_agentdojo.jsonl")
    if (os.path.exists(parsed_tau) and os.path.exists(parsed_dojo)
          and not args.fresh):
        print("Loading parsed JSONL artifacts ...")
        tau = read_jsonl(parsed_tau)
        dojo = read_jsonl(parsed_dojo)
    elif args.use_pickle_cache and os.path.exists(cache) and not args.fresh:
        import pickle
        print("Loading parsed pickle cache ...")
        with open(cache, "rb") as fh:
            tau, dojo = pickle.load(fh)
    else:
        require_raw_inputs()
        print("Parsing tau-bench ...")
        tau = parse_taubench()
        print(f"  {len(tau)} trajectories")
        print("Parsing AgentDojo (36k files, please wait) ...")
        dojo = parse_agentdojo(strict=not args.skip_invalid_json)
        print(f"  {len(dojo)} trajectories")
        if args.write_pickle_cache:
            import pickle
            with open(cache, "wb") as fh:
                pickle.dump((tau, dojo), fh)

    # write parsed datasets (drop bulky seq for compactness -> keep summary)
    with open(os.path.join(HERE, "parsed_taubench.jsonl"), "w") as fh:
        for r in tau:
            rr = {k: v for k, v in r.items() if k != "seq"}
            fh.write(json.dumps(rr, allow_nan=False) + "\n")
    with open(os.path.join(HERE, "parsed_agentdojo.jsonl"), "w") as fh:
        for r in dojo:
            rr = {k: v for k, v in r.items() if k != "seq"}
            fh.write(json.dumps(rr, allow_nan=False) + "\n")

    density_stats(tau, "tau-bench")
    density_stats(dojo, "agentdojo")
    density_stats([r for r in dojo if r["attack_type"] in (None, "none")],
                  "agentdojo_benign")
    burstiness_stats(tau, "tau-bench")
    burstiness_stats(dojo, "agentdojo")
    cross_agent_taubench(tau)
    shared_trigger_agentdojo(dojo)

    results["provenance"] = {
        "tau-bench": {
            "repo": "github.com/sierra-research/tau-bench",
            "path": "historical_trajectories/{gpt-4o,sonnet-35-new}-{retail,airline}.json",
            "n_trajectories": len(tau),
        },
        "agentdojo": {
            "repo": "github.com/ethz-spylab/agentdojo",
            "path": "runs/<model>/<suite>/<user_task>/<attack>/<injection>.json",
            "n_trajectories": len(dojo),
        },
    }
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, allow_nan=False)
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2, allow_nan=False))
    return tau, dojo

if __name__ == "__main__":
    main()
