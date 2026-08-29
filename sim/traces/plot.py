#!/usr/bin/env python3
"""Figure: real-trace evidence on correlated external-effect bursts."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def read_jsonl(path):
    rows = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL in {path}:{lineno}: {exc}") from exc
    return rows

def load_inputs(here=HERE):
    with open(os.path.join(here, "results.json")) as fh:
        results = json.load(fh)
    tau = read_jsonl(os.path.join(here, "parsed_taubench.jsonl"))
    dojo = read_jsonl(os.path.join(here, "parsed_agentdojo.jsonl"))
    return results, tau, dojo

def make_plot(results, tau, dojo, out=None):
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    # --- A: external-effect count per trajectory ---------------------------
    a = ax[0, 0]
    tc = [r["n_external"] for r in tau]
    dc = [r["n_external"] for r in dojo]
    overflow = 11
    bins = np.arange(0, overflow + 2) - 0.5
    tc_plot = [min(x, overflow) for x in tc]
    dc_plot = [min(x, overflow) for x in dc]
    a.hist(tc_plot, bins=bins, alpha=0.6, label=f"tau-bench (n={len(tc)})",
           density=True, color="#3b6ea5")
    a.hist(dc_plot, bins=bins, alpha=0.6, label=f"AgentDojo (n={len(dc)})",
           density=True, color="#c1544a")
    ticks = list(range(0, overflow + 1))
    a.set_xticks(ticks)
    a.set_xticklabels([str(x) for x in ticks[:-1]] + [f"{overflow}+"])
    a.set_xlabel("external effects per trajectory")
    a.set_ylabel("fraction of trajectories")
    a.set_title("(A) External-effect density (real traces)")
    a.legend(fontsize=7)

    # --- B: cross-agent concentration obs vs null, retail ------------------
    b = ax[0, 1]
    cc = results["cross_agent_taubench"]["retail"]["per_class_concentration"]
    classes = list(cc.keys())
    obs = [cc[c]["concentration_obs"] for c in classes]
    nul = [cc[c]["concentration_null"] for c in classes]
    x = np.arange(len(classes))
    b.bar(x-0.2, nul, 0.4, label="null (agent-idiosyncratic)", color="#b0b0b0")
    b.bar(x+0.2, obs, 0.4, label="observed (shared task)", color="#3b6ea5")
    b.set_xticks(x)
    b.set_xticklabels(classes, rotation=0, fontsize=8)
    b.set_ylim(0.5, 1.02)
    b.set_ylabel("fleet concentration  max(p,1-p)")
    b.set_title("(B) Same trigger -> same effect (tau-bench retail)")
    b.legend(fontsize=7, loc="lower right")
    b.axhline(0.5, ls=":", color="k", lw=0.6)

    # --- C: AgentDojo shared-trigger fleet response by suite ---------------
    c = ax[1, 0]
    shared = results["shared_trigger_agentdojo"]
    suite = shared.get("per_suite_attack_success",
                       shared.get("per_suite_injection_success", {}))
    order = ["slack", "banking", "travel", "workspace"]
    rates = [suite[s]["rate"] for s in order]
    c.bar(order, rates, color=["#c1544a", "#c1544a", "#c1544a", "#c1544a"])
    for i, s in enumerate(order):
        c.text(
            i, rates[i] + 0.01,
            f"{rates[i]*100:.0f}%\n"
            f"{suite[s]['successful_injections']}/{suite[s]['attacked_runs']}",
            ha="center", va="bottom", fontsize=7,
        )
    c.set_ylim(0, 0.55)
    c.set_ylabel("fleet fraction firing injected effect")
    c.set_title("(C) One planted instruction -> correlated\nirreversible burst (AgentDojo)")

    # --- D: temporal alignment of effects within trajectory ----------------
    d = ax[1, 1]
    rel_t = [p/(r["n_calls"]-1) for r in tau if r["n_calls"] > 1
             for p in r["ext_positions"]]
    rel_d = [p/(r["n_calls"]-1) for r in dojo if r["n_calls"] > 1
             for p in r["ext_positions"]]
    d.hist(rel_t, bins=20, alpha=0.6, density=True, color="#3b6ea5",
           label="tau-bench")
    d.hist(rel_d, bins=20, alpha=0.6, density=True, color="#c1544a",
           label="AgentDojo")
    d.set_xlabel("relative position of effect in trajectory")
    d.set_ylabel("density")
    d.set_title("(D) Effects cluster at trajectory end\n(aligned across agents)")
    d.legend(fontsize=7)

    fig.suptitle(
        "Real agent traces: external effects are shared-trigger-correlated across a fleet",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    out = os.path.join(HERE, "burstiness.png") if out is None else out
    fig.savefig(out, dpi=140, bbox_inches="tight")
    return out


def main():
    results, tau, dojo = load_inputs()
    out = make_plot(results, tau, dojo)
    print("wrote", out)


if __name__ == "__main__":
    main()
