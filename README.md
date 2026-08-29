# Irreversibility Budget Artifact

This directory contains the simulator, trace analysis, benchmark, and figure
generation code for the irreversibility-budget paper.

## Layout

- `sim/fleetsim.py`: discrete-event fleet simulator for RQ1-RQ5 and all arms.
- `sim/results/`: checked-in 300-run simulator outputs used by the paper.
- `sim/bench.py`: hierarchical escrow-ledger microbenchmark.
- `sim/bench_results.json`: checked-in benchmark output used by the paper.
- `sim/traces/`: real-trace classifier, analysis, parsed traces, and plot.
- `figures/plot_figs.py`: regenerates the three paper figure PDFs from
  `sim/results/results.json`.
- `figures/*.pdf`: generated outputs copied here for reproducibility. The paper
  tree keeps its own copies of the three PDFs that `main.tex` includes.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For tests, coverage, and dependency auditing:

```bash
pip install -r requirements-dev.txt
python -m coverage run -m pytest
python -m coverage report
python -m pip_audit -r requirements.txt
```

## Reproduce Outputs

From this directory:

```bash
python3 sim/fleetsim.py --runs 300 --out sim/results
python3 figures/plot_figs.py
python3 sim/bench.py --out sim/bench_results.json
python3 sim/traces/analyze.py
python3 sim/traces/plot.py
```

`sim/traces/analyze.py` uses the moved parsed JSONL trace artifacts by default.
Pickle cache loading is disabled unless `--use-pickle-cache` is passed, because
pickle is executable data. To re-parse raw cloned datasets, run with `--fresh`
and set either
`IRREVERSIBILITY_TRACE_RAW_ROOT` or both `IRREVERSIBILITY_TAUBENCH_DIR` and
`IRREVERSIBILITY_AGENTDOJO_DIR`.

## Trace Data Provenance

The parsed trajectories under `sim/traces/` are derived from two public
benchmarks, re-published here in parsed form so the analysis runs offline:

- tau-bench (retail + airline): github.com/sierra-research/tau-bench
- AgentDojo: github.com/ethz-spylab/agentdojo

See `sim/traces/results.json` → `provenance` for the exact paths used.

## Citation

If you use this artifact, please cite the paper:

> The Irreversibility Budget: Fleet-Level Risk Accounting and Admission
> Control for Agent Operating Systems. In Proceedings of the 2nd Workshop
> on OS Design for AI Agents (AgenticOS 2026).

## License

MIT — see [LICENSE](LICENSE).

## BibTeX

```bibtex
@inproceedings{Mohammadi2026IrreversibilityBudget,
  author    = {Mohammadi, Bardia and Bindschaedler, Laurent},
  title     = {The Irreversibility Budget: Fleet-Level Risk Accounting and Admission Control for Agent Operating Systems},
  booktitle = {Proceedings of the 2nd Workshop on {OS} Design for {AI} Agents ({AgenticOS} 2026)},
  year      = {2026},
  address   = {Prague, Czech Republic},
  publisher = {ACM},
  url       = {https://github.com/mpi-dsg/irreversibility-budget}
}
```
