# Context Retrieval Evaluation

ContextBench is available at: [https://huggingface.co/datasets/Schwerli/ContextBench](https://huggingface.co/datasets/Schwerli/ContextBench).

## Installation

```bash
# Recommended: install pinned runtime dependencies
pip install -r requirements.txt

# Critical: correct versions required (equivalent to requirements.txt)
pip install "tree-sitter==0.20.4" tree-sitter-languages
```

## Usage

```bash
python -m contextbench.evaluate \
    --gold <gold_path> \
    --pred <pred_path> \
    [--cache ./repos] \
    [--out results.jsonl]
```

## Example

```bash
cd <repo_root>

# Example with parquet gold (recommended)
python -m contextbench.evaluate \
    --gold data/full.parquet \
    --pred traj_verified/psf__requests-1142/psf__requests-1142.traj.json \
    --out result.jsonl
```

## Runner defaults (optional)

Some scripts support environment-variable defaults so you can relocate the repo:

```bash
export CONTEXTBENCH_GOLD=/path/to/full.parquet
export CONTEXTBENCH_CACHE=/path/to/repos_cache
export CONTEXTBENCH_SELECTED_CSV=/path/to/selected_500_instances.csv
export CONTEXTBENCH_RESULTS_ROOT=/path/to/results
export CONTEXTBENCH_TRAJ_AGENTLESS=/path/to/agentless_traj_root
export CONTEXTBENCH_TRAJ_MINISWE=/path/to/miniswe_traj_root
```

## Metrics

**Three Granularities**:
- **File**: File path sets
- **Symbol**: Definition nodes (class/function/method) within viewed spans
- **Span**: Byte intervals

**Final Context**: Coverage & Precision  
**Trajectory**: Per-step coverage, AUC-Coverage, Redundancy  
**EditLoc**: Edit localization recall & precision

## Output

```json
{
  "instance_id": "...",
  "num_steps": 6,
  "final": {
    "file": {"coverage": 1.0, "precision": 0.5, ...},
    "symbol": {"coverage": 1.0, "precision": 0.125, ...},
    "span": {"coverage": 1.0, "precision": 0.126, ...}
  },
  "trajectory": {
    "steps": [{"step": 1, "coverage": {...}}, ...],
    "auc_coverage": {"file": 1.0, "symbol": 1.0, "span": 1.0},
    "redundancy": {"file": 0.5, "symbol": 0.58, "span": 0.12}
  },
  "editloc": {"recall": 0.782, "precision": 1.0, ...}
}
```

## Directory Structure

```
contextbench/       # Python package
  core/             # Intervals, fileio, repo checkout
  parsers/          # Gold, trajectory, diff
  extractors/       # Tree-sitter symbol extraction
  metrics/          # Metric computation
  agents/           # Trajectory extractors
  evaluate.py       # Module entrypoint (python -m contextbench.evaluate)
  run_batch_eval_selected500.py  # Module entrypoint
docs/               # Documentation
```

## Agent extractors

See `docs/agents.md`.

## Results summary

This section summarizes the current contents of `results/**/all.jsonl`.

- **Macro average**: mean over instances (each instance counts equally).
- **Micro average**: aggregate intersections and sizes first, then divide (size-weighted). Micro is not shown in the leaderboard tables below.

### Main ContextBench leaderboard (LaTeX)

```latex
% \usepackage{multirow}  % <- make sure this is in the preamble
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lccccccccccccc}
\toprule
\multirow{2}{*}{Agent} &
\multicolumn{2}{c}{File-level} &
\multicolumn{2}{c}{Symbol-level} &
\multicolumn{2}{c}{Span-level} &
\multicolumn{2}{c}{EditLoc-level} &
\multicolumn{2}{c}{Context Size} &
\multirow{2}{*}{AUC-Cov$\uparrow$} &
\multirow{2}{*}{Redun.$\downarrow$} &
\multirow{2}{*}{Pass@1$\uparrow$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
& Cov.$\uparrow$ & Prec.$\uparrow$
& Cov.$\uparrow$ & Prec.$\uparrow$
& Cov.$\uparrow$ & Prec.$\uparrow$
& Cov.$\uparrow$ & Prec.$\uparrow$
& \#Files$\downarrow$ & \#Lines$\downarrow$
&  &  &  \\
\midrule
Agentless  & 0.656 & 0.398 & 0.357 & 0.393 & 0.056 & 0.791 & 0.599 & 0.599 & 4.044 & -- & 0.056 & 0.000 & -- \\
Prometheus & 0.799 & 0.346 & 0.716 & 0.255 & 0.655 & 0.202 & 0.812 & 0.812 & 5.945 & -- & 0.598 & 0.422 & -- \\
OpenHands  & --    & --    & --    & --    & --    & --    & --    & --    & --    & -- & --    & --    & -- \\
SWE-Agent  & 0.576 & 0.496 & 0.436 & 0.233 & 0.418 & 0.168 & 0.837 & 0.837 & 2.153 & -- & 0.563 & 0.094 & -- \\
\bottomrule
\end{tabular}
\vspace{2pt}
\caption{\textbf{Main ContextBench leaderboard.}
We report context coverage and precision at file/symbol/span/edit-location granularities,
together with retrieved context size, AUC-Coverage, redundancy, and end-to-end task success.}
\label{tab:contextbench_main}
\end{table*}
```

### Backbone model comparison (MiniSWE, LaTeX)

```latex
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccccccc}
\toprule
Backbone &
File Cov.$\uparrow$ & File Prec.$\uparrow$ & File F1$\uparrow$ &
Symbol Cov.$\uparrow$ & Symbol Prec.$\uparrow$ & Symbol F1$\uparrow$ &
Span Cov.$\uparrow$ & Span Prec.$\uparrow$ & Span F1$\uparrow$ &
AUC-Cov$\uparrow$ & Redun.$\downarrow$ \\
\midrule
GPT-5              & 0.472 & 0.464 & 0.468 & 0.424 & 0.597 & 0.496 & 0.402 & 0.559 & 0.468 & 0.322 & 0.078 \\
Claude 4.5 Sonnet  & 0.601 & 0.295 & 0.396 & 0.525 & 0.387 & 0.446 & 0.498 & 0.358 & 0.416 & 0.392 & 0.197 \\
Devstral 2         & 0.581 & 0.287 & 0.384 & 0.465 & 0.516 & 0.489 & 0.431 & 0.483 & 0.456 & 0.338 & 0.224 \\
Gemini 2.5 Pro     & 0.442 & 0.480 & 0.460 & 0.336 & 0.611 & 0.433 & 0.268 & 0.558 & 0.362 & 0.172 & 0.138 \\
\bottomrule
\end{tabular}
\vspace{2pt}
\caption{\textbf{Backbone model comparison.}
We instantiate the same agent setup (e.g., Mini SWE-agent) with different backbone models and evaluate on \benchmark.}
\label{tab:backbone_compare}
\end{table*}
```

### Notes

- `\#Files` is the macro-average of `final.file.pred_size` (average number of predicted files per instance).
- `\#Lines` and `Pass@1` are not currently recorded in `all.jsonl`, so they are shown as `--`.
