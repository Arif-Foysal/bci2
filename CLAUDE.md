# ieeg-ib Pipeline — Operating Guide

## Project

Zero-parameter, CPU-only seizure prediction framework using Information Theory (Mutual Information and Transfer Entropy) applied to intracranial EEG (iEEG) recordings from the HUP dataset (58 patients, NEMAR OpenNeuro ID: ds004100, also available as Kaggle dataset `mdariffaysalnayem/ds004100`). The project tests whether seizure onset is consistently preceded by a measurable collapse in network-wide Mutual Information — an "information bottleneck" — before physical voltage spikes occur, enabling causal, interpretable, and edge-deployable seizure prediction without any trained model. Four experimental phases: (1) temporal MI collapse detection via rolling Phi(t), (2) graph-theoretic network fragmentation, (3) MI vs. signal variance comparison, (4) Transfer Entropy SOZ identification via asymmetry index A_i.

## Architecture

- `src/ieeg_core.py` — core library: Kraskov MI estimator, rolling Phi(t), graph metrics, TE asymmetry index, preprocessing helpers. Uploaded to Kaggle as dataset `mdariffaysalnayem/ieeg-ib-core`
- `pipeline.py` — CLI for push/status/fetch/upload-src/upload-results
- `config.yaml` — username, datasets, phases, success criteria
- `templates/` — pristine notebook templates (checked in, never edited directly)
- `notebooks/` — working copies you edit (gitignored)
- `results/` — downloaded Kaggle outputs (gitignored)

## Environment

The Kaggle API token lives in `.env` (gitignored) and is loaded automatically by `pipeline.py`. Just activate the venv:

```bash
source .venv/bin/activate
python pipeline.py push notebooks/<phase>.py    # token loads from .env
```

If `.env` is missing, create it with `KAGGLE_API_TOKEN=KGAT_...` from kaggle.com/settings.

## Running an experiment

1. `python pipeline.py generate <phase>` (copies template to notebooks/)
2. Edit `notebooks/<phase>.py` if needed
3. `python pipeline.py push notebooks/<phase>.py`
4. `python pipeline.py wait <phase>` — single blocking call; prints one line per state change. Returns 0 on complete, non-zero on error/timeout.
5. `python pipeline.py fetch <phase>`
6. Read `results/<slug>/*.json` and grep `[RESULT]` lines. Use `python pipeline.py tail <phase>` for last 40 log lines — never cat the full log.

## Iteration protocol

When a notebook fails or success criteria aren't met:
- **Notebook bug** (syntax, shapes, logic) → edit `notebooks/<phase>.py`, re-push
- **Core math/lib bug** → edit `src/ieeg_core.py`, run `python pipeline.py upload-src`, **wait ~5 min**, re-push the notebook
- **Parameter issue** → tweak the notebook's config section, re-push
- **After 5 retries on the same notebook** → stop and ask the human

## Phase dependencies

When phase N's outputs feed phase N+1:
1. `python pipeline.py upload-results phaseN`
2. Uncomment the new dataset slug in `config.yaml` under `kaggle.datasets`
3. In `config.yaml` under the next notebook, add the key to its `datasets:` list

## Experimental phases and their notebooks

| Phase | Notebook | Key output | Success criterion |
|-------|----------|-----------|------------------|
| 1a | smoke test | `phase1a_results.json` | `experiment_complete == true` |
| 1b | rolling MI collapse | `phase1b_results.json` | `mean_tau_lead_seconds > 0` |
| 2a | graph fragmentation | `phase2a_results.json` | `network_fragmentation_detected == true` |
| 3a | MI vs. variance | `phase3a_results.json` | `pct_seizures_mi_leads_variance > 0.5` |
| 4a | TE SOZ identification | `phase4a_results.json` | `soz_auc > 0.5` |

## HUP dataset access inside Kaggle notebooks

The HUP iEEG dataset is mounted at a path under `/kaggle/input/`. Use `os.walk` to find BIDS files:

```python
import os
for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
    for f in filenames:
        if f.endswith("_ieeg.edf") or f.endswith("_ieeg.set"):
            print(os.path.join(dirpath, f))
```

BIDS structure: `sub-<id>/ses-<n>/ieeg/sub-<id>_ses-<n>_task-ictal_ieeg.edf`
SOZ annotations: `sub-<id>/sub-<id>_electrodes.tsv` — look for `soz` column.

## Output conventions (do not break)

- `core.format_result(key, value)` — emits a greppable `[RESULT]` line
- `core.save_results(dict, path)` — writes JSON to `/kaggle/working/`
- Plots saved as PNG to `/kaggle/working/`

## Token-cost discipline

- **Only three signals matter for "did this work?"**: `results/<slug>/*.json`, `[RESULT]` lines, and the kernel state from `wait`. Do not Read raw `.log` files; use `tail` if you need recent log lines.
- **Use `wait`, not a polling loop.** One `python pipeline.py wait <phase>` is dramatically cheaper in context tokens than running `status` repeatedly.
- **Keep CLAUDE.md, src/, templates/ stable across iterations.** They form the cacheable prefix.
- **Edit notebooks, not the CLI.**

## Gotchas

- **Dataset propagation lag ~5 min.** After `upload-src` or `upload-results`, wait before pushing.
- **`datasets create` silently no-ops** on existing datasets. The CLI handles this — don't bypass it.
- **Mount paths drift**: always `os.walk` to find `ieeg_core.py`.
- **GPU quota ~30h/week.** All phases use `gpu: false` — this project is CPU-only by design.
- **Auth**: `pipeline.py` loads `KAGGLE_API_TOKEN` from `.env` with `override=True`. If commands fail with 401/403, verify `.env` at repo root.

## When to stop and ask

- Phase boundary transitions (gate decisions on whether to advance)
- 5 retries exhausted
- Borderline / ambiguous results requiring scientific judgment
- Before any destructive actions (deleting kernels, force-pushing, etc.)
