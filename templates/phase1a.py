"""
ieeg-ib phase1a — smoke test and environment validation.

Validates the full pipeline end-to-end:
  - Core dataset mounts and imports correctly
  - [RESULT] lines are emitted
  - JSON results are saved to /kaggle/working/

Replace the body with real Phase 1 experiment code once the pipeline
is verified. See templates/phase1b.py for the real MI collapse experiment.

Outputs to /kaggle/working/:
  - phase1a_results.json
"""

import os
import sys

# Locate the core library inside Kaggle's input mount.
# Mount paths vary — always os.walk rather than hardcoding.
def _find_core(name):
    for dirpath, _, filenames in os.walk("/kaggle/input"):
        if f"{name}.py" in filenames:
            return dirpath
    return None

_core_path = _find_core("ieeg_core")
if _core_path is None:
    print("ERROR: ieeg_core.py not found under /kaggle/input/")
    print("Mounted files:")
    for dirpath, _, filenames in os.walk("/kaggle/input"):
        for f in filenames:
            print(" ", os.path.join(dirpath, f))
    raise ImportError("ieeg_core.py not found — did upload-src succeed?")
sys.path.insert(0, _core_path)

import ieeg_core as core

# ---------------------------------------------------------------------------
# Smoke test: verify core functions load and NumPy works
# ---------------------------------------------------------------------------

print("=" * 60)
print("PHASE 1A: smoke test")
print("=" * 60)

import numpy as np

rng = np.random.default_rng(42)

# Synthetic 2-channel MI test (known relationship)
t = np.linspace(0, 10, 5000)
ch1 = np.sin(2 * np.pi * 5 * t) + 0.3 * rng.standard_normal(5000)
ch2 = 0.8 * ch1 + 0.3 * rng.standard_normal(5000)   # correlated
ch3 = rng.standard_normal(5000)                         # independent

mi_correlated = core.kraskov_mi(ch1, ch2, k=5)
mi_independent = core.kraskov_mi(ch1, ch3, k=5)

print(f"\nSynthetic MI test:")
print(f"  MI(correlated pair):  {mi_correlated:.4f} nats  (expected > 0.3)")
print(f"  MI(independent pair): {mi_independent:.4f} nats  (expected ~0.0)")

assert mi_correlated > mi_independent, "MI estimator failed: correlated < independent"

# Synthetic rolling Phi(t) test
data_2ch = np.vstack([ch1, ch2])
phi, centers = core.rolling_global_mi(data_2ch, window_samples=500, step_samples=100, k=5)
print(f"\nRolling Phi(t) test:")
print(f"  Windows computed: {len(phi)}")
print(f"  Mean Phi: {phi.mean():.4f}")
print(f"  Std  Phi: {phi.std():.4f}")

results = {
    "experiment_complete": True,
    "mi_correlated": float(mi_correlated),
    "mi_independent": float(mi_independent),
    "mi_ratio": float(mi_correlated / (mi_independent + 1e-9)),
    "n_phi_windows": int(len(phi)),
    "mean_phi": float(phi.mean()),
    "std_phi": float(phi.std()),
    "python_version": sys.version,
    "numpy_version": np.__version__,
}

core.format_result("experiment_complete", results["experiment_complete"])
core.format_result("mi_correlated", f"{results['mi_correlated']:.4f}")
core.format_result("mi_independent", f"{results['mi_independent']:.4f}")
core.format_result("mi_ratio", f"{results['mi_ratio']:.2f}")
core.format_result("n_phi_windows", results["n_phi_windows"])

core.save_results(results, "/kaggle/working/phase1a_results.json")

print("\nSmoke test PASSED. Pipeline is operational.")
print("Next: python pipeline.py generate phase1b  (real MI collapse experiment)")
