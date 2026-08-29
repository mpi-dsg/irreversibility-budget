#!/usr/bin/env python3
"""Fleet simulator for the irreversibility budget feasibility study.

Simulates a tenant running N procurement agents over one or more budget
windows. Each agent proposes purchases; a correlated market trigger makes
"buy" locally rational for all agents at once. Admission regimes:

  local   -- per-call gates only: value cap + per-agent rate limit
            (every action individually well-governed, no shared state)
  budget  -- the irreversibility budget: a tenant-level ledger that
            reserves the declared charge c(e) (a gamma-quantile of the
            effect's residual-loss distribution) and denies admission
            when reserved+committed charge would exceed the budget B
  static  -- per-agent partitions of B (no shared ledger); the static
            multiplier m gives each agent m*B/N

Ground truth: each executed purchase realizes a residual exposure
v * (1 - recovery). The tenant overdraws when realized unapproved
exposure exceeds its per-window risk tolerance R.

Experiments:
  E1  composition failure: P(overdraw) and exposure distribution,
      local vs budget, plus a fragmentation attack variant
  E2  safety-liveness tradeoff: sweep budget size B/R
  E3  mispricing sensitivity: scale declared charges by epsilon
  E4  allocation policy: shared ledger vs static per-agent partitions
  E5  replenishment: campaign adversary across windows, with and
      without a longer-horizon campaign ledger
  E6  re-authorization: approval capacity H restores starved traffic
  E7  risk typing: typed per-class charges vs an untyped face-value
      spend cap on a mixed refundable/final workload

Usage: python3 fleetsim.py [--runs 300] [--out results/]
"""

import argparse
import csv
import json
import math
import os

import numpy as np

# ---------------------------------------------------------------- parameters

P = dict(
    n_agents=50,
    ticks=1000,              # one budget window
    p_base=0.0005,           # per-tick proposal probability, calm regime
                             # (routine window demand ~87% of B: business
                             # as usual fits the budget; the burst does not)
    p_burst=0.05,            # per-tick proposal probability during the burst
    burst_len=50,            # ticks of correlated "buy" regime
    value_mu=9.6, value_sigma=0.5,   # lognormal purchase value, ~$15k median
    value_cap=50_000.0,      # per-call gate: max single purchase
    rate_limit=20,           # per-agent purchases per window
    recovery_a=6.0, recovery_b=2.0,  # Beta recovery fraction, mean 0.75
    tolerance=250_000.0,     # tenant risk tolerance R per window
    gamma=0.95,              # confidence level for the declared charge
    frag_total=1_500_000.0,  # fragmentation attack: total value to move
    frag_piece=10_000.0,     # piece size, below value_cap
    frag_agents=10,          # colluding agents the attack is spread over
    campaign_total=1_350_000.0,  # campaign attack value across all windows
    # mixed workload (E7): refundable vs final purchases
    ref_a=18.0, ref_b=2.0,   # refundable recovery Beta, mean 0.90
    fin_a=1.5, fin_b=6.0,    # final recovery Beta, mean 0.20
    # correlated recoverability (E8): the same latent driver that raises
    # buy propensity also depresses recovery during the burst
    corr_a=2.0, corr_b=6.0,  # burst-time recovery Beta, mean 0.25
)

VALID_REGIMES = {"local", "budget", "static", "cap", "breaker"}
VALID_ATTACKS = {None, "asap", "campaign"}
VALID_WORKLOADS = {"single", "mixed"}
VALID_CHARGING = {"typed", "pooled", "face"}
MAX_RUNS_WITH_UNIQUE_SWEEP_SEEDS = 500


def _is_int_like(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _require_finite(name, value, *, minimum=None, exclusive_minimum=False):
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        if exclusive_minimum and value <= minimum:
            raise ValueError(f"{name} must be > {minimum}")
        if not exclusive_minimum and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    return value


def _validate_simulation_inputs(regime, attack, workload, charging, seed, windows,
                                budget_frac, eps, static_mult, campaign_budget,
                                reauth, fair_share, n_workflows,
                                workflow_budget_frac, detect_lag, detect_k,
                                ref_recovery, n_agents, p_burst_mult, gamma,
                                corr_b, value_sigma, breaker_delay,
                                frag_agents):
    if regime not in VALID_REGIMES:
        raise ValueError(f"unknown admission regime: {regime!r}")
    if attack not in VALID_ATTACKS:
        raise ValueError(f"unknown attack mode: {attack!r}")
    if workload not in VALID_WORKLOADS:
        raise ValueError(f"unknown workload: {workload!r}")
    if charging not in VALID_CHARGING:
        raise ValueError(f"unknown charging mode: {charging!r}")
    if not _is_int_like(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not _is_int_like(windows) or windows < 1:
        raise ValueError("windows must be a positive integer")
    if not _is_int_like(n_workflows) or n_workflows < 1:
        raise ValueError("n_workflows must be a positive integer")
    if not _is_int_like(detect_lag) or detect_lag < 0:
        raise ValueError("detect_lag must be a non-negative integer")
    if n_agents is not None and (not _is_int_like(n_agents) or n_agents < 1):
        raise ValueError("n_agents must be a positive integer")
    if frag_agents is not None and (not _is_int_like(frag_agents) or frag_agents < 1):
        raise ValueError("frag_agents must be a positive integer")
    if breaker_delay is not None and (
        not _is_int_like(breaker_delay) or breaker_delay < 0
    ):
        raise ValueError("breaker_delay must be a non-negative integer")
    _require_finite("budget_frac", budget_frac, minimum=0.0)
    _require_finite("eps", eps, minimum=0.0)
    _require_finite("static_mult", static_mult, minimum=0.0)
    _require_finite("reauth", reauth, minimum=0.0)
    _require_finite("detect_k", detect_k, minimum=0.0)
    if campaign_budget is not None:
        _require_finite("campaign_budget", campaign_budget, minimum=0.0)
    if fair_share is not None:
        _require_finite("fair_share", fair_share, minimum=0.0)
    if workflow_budget_frac is not None:
        _require_finite("workflow_budget_frac", workflow_budget_frac, minimum=0.0)
    if p_burst_mult is not None:
        _require_finite("p_burst_mult", p_burst_mult, minimum=0.0)
    if gamma is not None:
        _require_finite("gamma", gamma, minimum=0.0, exclusive_minimum=True)
        if gamma >= 1.0:
            raise ValueError("gamma must be < 1.0")
    if corr_b is not None:
        _require_finite("corr_b", corr_b, minimum=0.0, exclusive_minimum=True)
    if value_sigma is not None:
        _require_finite("value_sigma", value_sigma, minimum=0.0)
    if ref_recovery is not None:
        _require_finite("ref_recovery", ref_recovery, minimum=0.0,
                        exclusive_minimum=True)
        if ref_recovery >= 1.0:
            raise ValueError("ref_recovery must be < 1.0")


def _residual_quantile(gamma, a=None, b=None, params=None):
    """Residual-loss factor at confidence gamma: the declared charge is
    c(e) = v * this quantile of (1 - recovery)."""
    params = P if params is None else params
    a = params["recovery_a"] if a is None else a
    b = params["recovery_b"] if b is None else b
    try:
        from scipy.stats import beta as beta_dist
        return 1.0 - beta_dist.ppf(1.0 - gamma, a, b)
    except ImportError:  # numerical inverse via sampling
        rng = np.random.default_rng(0)
        s = 1.0 - rng.beta(a, b, 200_000)
        return float(np.quantile(s, gamma))


def _pooled_quantile(gamma, ref_a, ref_b, params=None):
    """0.95 residual quantile of the 50/50 refundable+final mixture: the
    charge a distribution-aware but untyped cap would use (E7 M3 baseline)."""
    params = P if params is None else params
    rng = np.random.default_rng(12345)
    m = 200_000
    pick = rng.random(m) < 0.5
    s = np.where(pick,
                 1.0 - rng.beta(params["fin_a"], params["fin_b"], m),
                 1.0 - rng.beta(ref_a, ref_b, m))
    return float(np.quantile(s, gamma))


def simulate(seed, regime, budget_frac=1.0, eps=1.0, attack=None,
             static_mult=1.0, windows=1, campaign_budget=None, reauth=0,
             workload="single", charging="typed", correlated_burst=False,
             frag_charge_factor=None, stress_charging=False,
             fair_share=None, n_workflows=1, workflow_budget_frac=None,
             rogue_workflow=False, detector=False, detect_lag=0,
             detect_k=3.0, ref_recovery=None, n_agents=None,
             p_burst_mult=None, gamma=None, corr_b=None, value_sigma=None,
             breaker_delay=None, frag_agents=None):
    """Run `windows` consecutive budget windows. Returns outcome metrics.

    regime: 'local', 'budget', or 'static'
    budget_frac: B = budget_frac * tolerance (per window)
    eps: declared-charge scaling (mispricing)
    attack: None, 'asap' (drip fragments every tick from t=0), or
            'campaign' (pace fragments evenly across the whole horizon)
    correlated_burst: burst-time effects draw recovery from the depressed
            corr_a/corr_b distribution while charges stay at the declared
            calm-time quantile (the shared latent driver of Sec. 6)
    frag_charge_factor: misdeclaration attack: fragments are charged at
            this residual factor (e.g. the refundable-class quantile)
            while realizing calm-time recovery
    stress_charging: the runtime detects the burst regime and charges
            burst-time effects at the burst-calibrated quantile (a first
            correlation-aware policy; detection is assumed perfect)
    static_mult: per-agent budget = static_mult * B / N (regime 'static')
    campaign_budget: optional non-resetting ledger cap across windows
    reauth: authority approvals per window for denied effects, processed
            at a uniform rate with a bounded backlog (token bucket)
    fair_share: per-principal reservation cap as a fraction of B, layered
            on the shared ledger; bounds any one agent's share so a
            collusion cannot seize the whole budget ahead of honest work
    n_workflows: agents partition into this many workflows under the tenant
    workflow_budget_frac: per-workflow sub-budget as a fraction of B (the
            hierarchical ledger); None means a flat tenant ledger only
    rogue_workflow: the correlated burst hits only workflow 0's agents,
            so the sibling workflow's honest traffic is the bystander
    detector: signal-driven correlation-aware charging. The runtime does
            not see phase; it infers pressure from the admitted-proposal
            rate exceeding detect_k times its trailing calm mean, and
            after detect_lag ticks raises the charge from q to q_corr
    detect_lag: detection delay in ticks before the elevated charge applies
    detect_k: proposal-rate multiple over the calm baseline that trips it
    ref_recovery: override the refundable-class mean recovery (E7 M2 sweep)
    n_agents, p_burst_mult, gamma, corr_b, value_sigma: robustness overrides
            (fleet size, burst intensity multiplier, confidence level,
            burst-recovery Beta shape, value-distribution tail); None keeps
            the default in P
    breaker_delay: regime 'breaker' halts the fleet once realized exposure
            observed with this tick lag reaches R (a reactive circuit breaker)
    """
    frag_agents = P["frag_agents"] if frag_agents is None else frag_agents
    _validate_simulation_inputs(
        regime, attack, workload, charging, seed, windows, budget_frac, eps,
        static_mult, campaign_budget, reauth, fair_share, n_workflows,
        workflow_budget_frac, detect_lag, detect_k, ref_recovery, n_agents,
        p_burst_mult, gamma, corr_b, value_sigma, breaker_delay, frag_agents,
    )
    rng = np.random.default_rng(seed)
    frag_rng = np.random.default_rng([seed, 999983])  # attack draws off the
    #   main stream, so attack and no-attack arms share the honest sequence
    #   (common random numbers stay paired; see mc() CIs)
    cfg = dict(P)
    cfg["frag_agents"] = frag_agents
    n = n_agents if n_agents is not None else cfg["n_agents"]
    if attack and frag_agents > n:
        raise ValueError("frag_agents must be <= n_agents for attack simulations")
    T = cfg["ticks"]
    gamma_v = gamma if gamma is not None else cfg["gamma"]
    corr_b_v = corr_b if corr_b is not None else cfg["corr_b"]
    sigma_v = value_sigma if value_sigma is not None else cfg["value_sigma"]
    pburst_v = cfg["p_burst"] * (p_burst_mult if p_burst_mult is not None else 1.0)
    if pburst_v > 1.0:
        raise ValueError("p_burst * p_burst_mult must be <= 1.0")
    R = cfg["tolerance"]
    B = budget_frac * R
    q = _residual_quantile(gamma_v, params=cfg)
    ref_a, ref_b = cfg["ref_a"], cfg["ref_b"]
    if ref_recovery is not None:      # M2 sweep: reshape refundable class
        ref_a, ref_b = ref_recovery * 20.0, (1.0 - ref_recovery) * 20.0
    q_ref = _residual_quantile(gamma_v, ref_a, ref_b)
    q_fin = _residual_quantile(gamma_v, cfg["fin_a"], cfg["fin_b"])
    q_corr = _residual_quantile(gamma_v, cfg["corr_a"], corr_b_v)
    q_pool = _pooled_quantile(gamma_v, ref_a, ref_b, params=cfg)  # M3
    workflow_of = (np.arange(n) * n_workflows) // n
    colluder_set = set(range(frag_agents)) if attack else set()

    # burst occurs once, in the first window
    burst_start = rng.integers(100, T - cfg["burst_len"] - 100)
    burst_end = burst_start + cfg["burst_len"]

    frag_total = cfg["campaign_total"] if attack == "campaign" else cfg["frag_total"]
    colluders = list(range(frag_agents)) if attack else []
    frag_next = 0
    frag_moved = 0.0
    horizon = windows * T

    campaign_reserved = 0.0
    approved_exposure = 0.0
    exposure_windows = []        # realized unapproved exposure per window
    exec_windows = []            # routine value executed / proposed per window
    approved_charge = 0.0
    n_exec = n_denied = n_reauth = n_honest_denied = 0
    proposed = {"pre": 0.0, "burst": 0.0, "post": 0.0}
    executed = {"pre": 0.0, "burst": 0.0, "post": 0.0}
    sibling_prop = sibling_exec = 0.0   # non-rogue workflow (hierarchy test)

    for w in range(windows):
        reserved = 0.0           # per-window ledger, resets at window roll
        exposure_w = 0.0
        reauth_tokens = 0.0      # authority capacity accrues at reauth/T
        rate_used = np.zeros(n, dtype=int)
        agent_reserved = np.zeros(n)   # per-agent reserved (static / fair-share)
        wf_reserved = np.zeros(n_workflows)  # per-workflow sub-ledgers
        prop_w = exec_w = 0.0
        detect_hits = 0          # consecutive above-threshold ticks
        boost_ticks = 0          # ticks of elevated charge remaining
        exposure_by_tick = np.zeros(T)   # breaker: realized exposure per tick
        halted = False           # breaker tripped

        for t in range(T):
            g = w * T + t        # global tick
            if regime == "breaker":  # reactive halt on delayed-observed loss
                obs_t = t - (breaker_delay or 0)
                if obs_t > 0 and exposure_by_tick[:obs_t].sum() >= R:
                    halted = True
            if reauth:
                reauth_tokens = min(reauth_tokens + reauth / T,
                                    max(1.0, reauth / 10))
            if w > 0:
                phase = "post"
            elif t < burst_start:
                phase = "pre"
            elif t < burst_end:
                phase = "burst"
            else:
                phase = "post"
            draws = rng.random(n)
            if rogue_workflow:   # burst confined to workflow 0's agents
                p_vec = np.where((workflow_of == 0) & (phase == "burst"),
                                 pburst_v, cfg["p_base"])
                proposers = list(np.where(draws < p_vec)[0])
            else:
                p = pburst_v if phase == "burst" else cfg["p_base"]
                proposers = list(np.where(draws < p)[0])

            # signal-driven detector: infer pressure from the proposal-rate
            # spike, not from the (hidden) phase label; charge boost lags
            if detector:
                if len(proposers) > detect_k * cfg["p_base"] * n:
                    detect_hits += 1
                    if detect_hits > detect_lag:
                        boost_ticks = cfg["burst_len"]
                else:
                    detect_hits = 0
                boost_ticks = max(0, boost_ticks - 1)

            # one fragment per eligible tick from the collusion
            frag_proposer = -1
            want_frag = False
            if attack == "asap":
                want_frag = frag_moved + cfg["frag_piece"] <= frag_total
            elif attack == "campaign":
                pace = frag_total * (g + 1) / horizon
                want_frag = frag_moved + cfg["frag_piece"] <= pace
            if want_frag:
                frag_proposer = colluders[frag_next % len(colluders)]
                frag_next += 1
                proposers.append(-1)  # sentinel: the fragment proposal

            for i in proposers:
                is_frag = i == -1
                if is_frag:
                    i, v = frag_proposer, cfg["frag_piece"]
                else:
                    v = float(rng.lognormal(cfg["value_mu"], sigma_v))

                # per-call gate: all regimes enforce it (the status quo)
                if v > cfg["value_cap"] or rate_used[i] >= cfg["rate_limit"]:
                    continue
                if not is_frag:
                    proposed[phase] += v
                    prop_w += v
                    if rogue_workflow and workflow_of[i] != 0:
                        sibling_prop += v

                # mixed workload: half the purchases are mostly refundable,
                # half are mostly final; a typed ledger charges each class
                # its own quantile, an untyped cap charges face value (or the
                # pooled quantile when the cap is distribution-aware, M3)
                if workload == "mixed" and not is_frag:
                    is_final = rng.random() < 0.5
                    rec_a = cfg["fin_a"] if is_final else ref_a
                    rec_b = cfg["fin_b"] if is_final else ref_b
                    if charging == "typed":
                        q_eff = q_fin if is_final else q_ref
                    elif charging == "pooled":
                        q_eff = q_pool
                    else:
                        q_eff = 1.0
                else:
                    rec_a, rec_b = cfg["recovery_a"], cfg["recovery_b"]
                    q_eff = q

                charge = eps * v * q_eff
                if stress_charging and phase == "burst":
                    charge = eps * v * q_corr    # perfect-detection upper bound
                if detector and boost_ticks > 0:
                    charge = eps * v * q_corr    # signal-driven, lagged
                if is_frag and frag_charge_factor is not None:
                    charge = eps * v * frag_charge_factor
                wf = workflow_of[i]
                if regime == "cap":     # untyped aggregate spend cap
                    charge = eps * v * (q_pool if charging == "pooled" else 1.0)
                    denied = reserved + charge > B
                elif regime == "budget":
                    denied = reserved + charge > B
                elif regime == "static":
                    denied = agent_reserved[i] + charge > static_mult * B / n
                elif regime == "breaker":   # reactive: execute until halted
                    denied = halted
                else:
                    denied = False
                if not denied and fair_share is not None:
                    denied = agent_reserved[i] + charge > fair_share * B
                if not denied and workflow_budget_frac is not None:
                    denied = wf_reserved[wf] + charge > workflow_budget_frac * B
                if not denied and campaign_budget is not None:
                    denied = campaign_reserved + charge > campaign_budget

                approved = False
                if denied and reauth_tokens >= 1.0:
                    reauth_tokens -= 1.0
                    n_reauth += 1
                    approved = True   # authority extension, charge tracked
                elif denied:
                    n_denied += 1
                    if not is_frag and i not in colluder_set:
                        n_honest_denied += 1
                    continue

                # execute
                rate_used[i] += 1
                if correlated_burst and phase == "burst":
                    rec_a, rec_b = cfg["corr_a"], corr_b_v
                r = (frag_rng if is_frag else rng).beta(rec_a, rec_b)
                x = v * (1.0 - r)
                exposure_by_tick[t] += x
                if approved:
                    approved_charge += charge
                    approved_exposure += x
                else:
                    exposure_w += x
                    reserved += charge
                    wf_reserved[wf] += charge
                    if campaign_budget is not None:
                        campaign_reserved += charge
                    if regime == "static" or fair_share is not None:
                        agent_reserved[i] += charge
                n_exec += 1
                if is_frag:
                    frag_moved += v
                else:
                    executed[phase] += v
                    exec_w += v
                    if rogue_workflow and wf != 0:
                        sibling_exec += v

        exposure_windows.append(exposure_w)
        exec_windows.append((exec_w / prop_w) if prop_w else 1.0)

    total_prop = sum(proposed.values())
    total_exec = sum(executed.values())
    return dict(
        overdraw=int(max(exposure_windows) > R),
        exposure=sum(exposure_windows),
        n_exec=n_exec,
        n_denied=n_denied,
        n_honest_denied=n_honest_denied,
        n_reauth=n_reauth,
        approved_charge=approved_charge,
        approved_exposure=approved_exposure,
        utility_frac=(total_exec / total_prop) if total_prop else 1.0,
        exec_pre=(executed["pre"] / proposed["pre"]) if proposed["pre"] else 1.0,
        exec_burst=(executed["burst"] / proposed["burst"]) if proposed["burst"] else 1.0,
        exec_post=(executed["post"] / proposed["post"]) if proposed["post"] else 1.0,
        exec_w0=exec_windows[0],
        exec_wlast=exec_windows[-1],
        sibling_exec=(sibling_exec / sibling_prop) if sibling_prop else 1.0,
        frag_moved=frag_moved,
    )


def mc(runs, seed0, **kw):
    """Monte Carlo: aggregate mean and 95% CI over runs."""
    if not _is_int_like(runs) or runs < 1:
        raise ValueError("runs must be a positive integer")
    rows = [simulate(seed0 + k, **kw) for k in range(runs)]
    out = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = float(vals.mean())
        if len(vals) == 1:
            out[key + "_ci"] = 0.0
        else:
            out[key + "_ci"] = float(
                1.96 * vals.std(ddof=1) / math.sqrt(len(vals))
            )
    out["exposure_p95"] = float(np.quantile(
        np.array([r["exposure"] for r in rows]), 0.95))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be positive")
    if args.runs > MAX_RUNS_WITH_UNIQUE_SWEEP_SEEDS:
        ap.error(
            f"--runs must be <= {MAX_RUNS_WITH_UNIQUE_SWEEP_SEEDS}; several "
            "experiment seed blocks are 500 seeds apart"
        )
    os.makedirs(args.out, exist_ok=True)
    R = P["tolerance"]

    results = {}

    def run(key, seed0, **kw):
        m = mc(args.runs, seed0, **kw)
        results[key] = m
        return m

    # E1: composition failure and fix (+ fragmentation attack)
    print("== E1: composition failure ==")
    for name, kw in [
        ("local", dict(regime="local")),
        ("budget", dict(regime="budget")),
        ("cap", dict(regime="cap")),
        ("local_attack", dict(regime="local", attack="asap")),
        ("budget_attack", dict(regime="budget", attack="asap")),
        ("budget_attack_misdecl", dict(regime="budget", attack="asap",
                                       frag_charge_factor=0.226)),
    ]:
        m = run("e1_" + name, 1000, **kw)
        print(f"  {name:14s} P(overdraw)={m['overdraw']:.3f}"
              f"  mean exposure=${m['exposure']:,.0f} (R=${R:,.0f})"
              f"  p95=${m['exposure_p95']:,.0f}"
              f"  denied={m['n_denied']:.1f}"
              f"  frag_moved=${m['frag_moved']:,.0f}")

    # E2: safety-liveness tradeoff, sweep B/R
    print("== E2: budget sweep ==")
    for bf in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        m = run(f"e2_bf{bf}", 2000, regime="budget", budget_frac=bf)
        print(f"  B/R={bf:4.2f}  P(overdraw)={m['overdraw']:.3f}"
              f"  utility_frac={m['utility_frac']:.3f}"
              f"  pre/burst/post={m['exec_pre']:.2f}/{m['exec_burst']:.2f}/{m['exec_post']:.2f}"
              f"  denied={m['n_denied']:.1f}")

    # E3: mispricing sensitivity, scale charges by eps
    print("== E3: mispricing ==")
    for eps in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]:
        m = run(f"e3_eps{eps}", 3000, regime="budget", eps=eps)
        print(f"  eps={eps:4.2f}  P(overdraw)={m['overdraw']:.3f}"
              f"  utility_frac={m['utility_frac']:.3f}"
              f"  denied={m['n_denied']:.1f}")

    # E4: allocation policy — shared ledger vs static per-agent partitions
    print("== E4: allocation ==")
    for name, kw in [
        ("static1", dict(regime="static", static_mult=1.0)),
        ("static10", dict(regime="static", static_mult=10.0)),
    ]:
        m = run("e4_" + name, 4000, **kw)
        print(f"  {name:9s} P(overdraw)={m['overdraw']:.3f}"
              f"  exposure=${m['exposure']:,.0f}"
              f"  routine pre={m['exec_pre']:.2f}"
              f"  utility={m['utility_frac']:.3f}")

    # E5: replenishment — campaign adversary across 3 windows
    print("== E5: campaign across windows ==")
    for name, kw in [
        ("no_attack", dict(regime="budget", windows=3)),
        ("no_attack_ledger", dict(regime="budget", windows=3,
                                  campaign_budget=1.5 * R)),
        ("window_only", dict(regime="budget", windows=3, attack="campaign")),
        ("campaign_ledger", dict(regime="budget", windows=3, attack="campaign",
                                 campaign_budget=1.5 * R)),
        ("campaign_ledger3R", dict(regime="budget", windows=3,
                                   attack="campaign",
                                   campaign_budget=3.0 * R)),
    ]:
        m = run("e5_" + name, 5000, **kw)
        print(f"  {name:16s} frag_moved=${m['frag_moved']:,.0f}"
              f"  of ${P['campaign_total']:,.0f}"
              f"  P(overdraw)={m['overdraw']:.3f}"
              f"  routine w0={m['exec_w0']:.2f} wlast={m['exec_wlast']:.2f}")

    # E7: risk typing vs untyped caps (face-value and pooled-quantile),
    # plus an M2 sweep of how the typing gap depends on class separation
    print("== E7: typed charges vs untyped cap ==")
    for name, kw in [
        ("typed", dict(regime="budget", workload="mixed", charging="typed")),
        ("face_cap", dict(regime="cap", workload="mixed", charging="face")),
        ("pooled_cap", dict(regime="cap", workload="mixed", charging="pooled")),
    ]:
        m = run("e7_" + name, 7000, **kw)
        print(f"  {name:11s} P(overdraw)={m['overdraw']:.3f}"
              f"  utility_frac={m['utility_frac']:.3f}"
              f"  routine pre={m['exec_pre']:.2f}"
              f"  exposure=${m['exposure']:,.0f}")
    base = results["e7_face_cap"]["utility_frac"]
    for rm in [0.90, 0.75, 0.60, 0.50]:
        m = run(f"e7_sep{rm}", 7500, regime="budget", workload="mixed",
                charging="typed", ref_recovery=rm)
        print(f"  refundable mean {rm:.2f}: typed util={m['utility_frac']:.3f}"
              f"  gap vs face cap = {m['utility_frac']/base:.2f}x")

    # E8: correlated recoverability breaks per-effect quantile charging,
    # and a signal-driven lagged detector partly buys it back
    print("== E8: correlated recoverability ==")
    for name, kw in [
        ("budget_corr", dict(regime="budget", correlated_burst=True)),
        ("local_corr", dict(regime="local", correlated_burst=True)),
        ("budget_corr_stress", dict(regime="budget", correlated_burst=True,
                                    stress_charging=True)),
    ]:
        m = run("e8_" + name, 8000, **kw)
        print(f"  {name:20s} P(overdraw)={m['overdraw']:.3f}"
              f"  mean exposure=${m['exposure']:,.0f}"
              f"  burst admitted={m['exec_burst']:.3f}")
    for L in [0, 5, 10, 25]:
        m = run(f"e8_detect_L{L}", 8500, regime="budget", correlated_burst=True,
                detector=True, detect_lag=L)
        print(f"  detector lag {L:2d}: P(overdraw)={m['overdraw']:.3f}"
              f"  mean exposure=${m['exposure']:,.0f}"
              f"  burst admitted={m['exec_burst']:.3f}")

    # E9: fair-share admission bounds a colluder's share of the ledger
    print("== E9: fair-share admission ==")
    for name, kw in [
        ("fcfs", dict(regime="budget", attack="asap")),
        ("fair", dict(regime="budget", attack="asap", fair_share=0.05)),
    ]:
        m = run("e9_" + name, 9000, **kw)
        print(f"  {name:5s} frag_moved=${m['frag_moved']:,.0f}"
              f"  honest_denied={m['n_honest_denied']:.0f}"
              f"  P(overdraw)={m['overdraw']:.3f}")

    # E10: hierarchical workflow ledger contains a rogue workflow
    print("== E10: hierarchical ledgers ==")
    for name, kw in [
        ("flat", dict(regime="budget", n_workflows=2, rogue_workflow=True)),
        ("hier", dict(regime="budget", n_workflows=2, rogue_workflow=True,
                      workflow_budget_frac=0.6)),
    ]:
        m = run("e10_" + name, 10000, **kw)
        print(f"  {name:5s} sibling_exec={m['sibling_exec']:.3f}"
              f"  P(overdraw)={m['overdraw']:.3f}")

    # E6: re-authorization capacity restores starved traffic
    print("== E6: re-authorization ==")
    for h in [0, 10, 50]:
        m = run(f"e6_h{h}", 6000, regime="budget", reauth=h)
        print(f"  H={h:3d}  exec_post={m['exec_post']:.2f}"
              f"  approved_charge=${m['approved_charge']:,.0f}"
              f"  approved_exposure=${m['approved_exposure']:,.0f}"
              f"  P(silent overdraw)={m['overdraw']:.3f}"
              f"  reauth_used={m['n_reauth']:.1f}")

    # E11 (Tier 1): fleet-size scaling — the composition failure grows with N,
    # the ledger holds regardless (tenant tolerance R fixed)
    print("== E11: fleet-size scaling ==")
    for N in [10, 50, 200, 1000]:
        ml = run(f"e11_local_n{N}", 11000, regime="local", n_agents=N)
        mb = run(f"e11_budget_n{N}", 11500, regime="budget", n_agents=N)
        print(f"  N={N:4d}  local exp={ml['exposure']/R:.2f}R (P={ml['overdraw']:.2f})"
              f"  |  budget exp={mb['exposure']/R:.2f}R (P={mb['overdraw']:.2f})")

    # E12 (Tier 1): robustness — qualitative results survive parameter changes
    print("== E12: robustness sweeps ==")
    for label, kw in [
        ("burst x0.5", dict(regime="budget", p_burst_mult=0.5)),
        ("burst x2", dict(regime="budget", p_burst_mult=2.0)),
        ("gamma 0.90", dict(regime="budget", gamma=0.90)),
        ("gamma 0.99", dict(regime="budget", gamma=0.99)),
        ("heavy tail", dict(regime="budget", value_sigma=1.0)),
    ]:
        m = run("e12_" + label.replace(" ", "_"), 12000, **kw)
        # corr robustness: does additive charging still break as coupling varies
        print(f"  {label:11s} P(overdraw)={m['overdraw']:.3f}"
              f"  exp={m['exposure']/R:.2f}R  util={m['utility_frac']:.3f}")
    for cb in [4.0, 6.0, 8.0]:   # burst-recovery Beta b: higher = worse recovery
        m = run(f"e12_corr_b{cb}", 12500, regime="budget",
                correlated_burst=True, corr_b=cb)
        mean_rec = P["corr_a"] / (P["corr_a"] + cb)
        print(f"  corr mean-recovery {mean_rec:.2f}: P(overdraw)={m['overdraw']:.3f}")

    # E13 (Tier 2): reactive circuit breaker halts on realized loss — too late
    # for irreversible effects, which commit before their loss is observed
    print("== E13: circuit breaker vs budget ==")
    mbud = run("e13_budget", 13000, regime="budget")
    print(f"  budget         exp={mbud['exposure']/R:.2f}R  P(overdraw)={mbud['overdraw']:.2f}")
    for d in [0, 10, 50]:
        m = run(f"e13_breaker_d{d}", 13500 + d, regime="breaker", breaker_delay=d)
        print(f"  breaker lag {d:2d}   exp={m['exposure']/R:.2f}R  P(overdraw)={m['overdraw']:.2f}")

    # E14 (Tier 2): adaptive adversary — fair share bounds each principal, so
    # more colluders raise the take linearly (motivates attribution, Sec. 6)
    print("== E14: adaptive adversary vs fair share ==")
    for na in [10, 20, 40]:
        m = run(f"e14_colluders{na}", 14000 + na, regime="budget",
                attack="asap", fair_share=0.05, frag_agents=na)
        print(f"  colluders={na:3d}  frag_moved=${m['frag_moved']:,.0f}"
              f"  honest_denied={m['n_honest_denied']:.0f}")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"params": P, "runs": args.runs, "results": results}, f)

    # flat CSV for the figure script
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = ["overdraw", "overdraw_ci", "exposure", "exposure_ci",
                "exposure_p95", "utility_frac", "utility_frac_ci",
                "exec_pre", "exec_burst", "exec_post",
                "exec_w0", "exec_wlast",
                "n_denied", "n_honest_denied", "n_reauth", "approved_charge",
                "approved_exposure", "sibling_exec", "frag_moved"]
        w.writerow(["config"] + keys)
        for cfg, m in results.items():
            w.writerow([cfg] + [m[k] for k in keys])
    print(f"wrote {args.out}/results.json and summary.csv")


if __name__ == "__main__":
    main()
