"""
ieeg-ib core library.

Uploaded as a Kaggle dataset by `pipeline.py upload-src`. All notebooks
import this module after locating it under /kaggle/input/.

Provides:
  - Result saving conventions (format_result, save_results)
  - Kraskov k-NN mutual information estimator
  - Rolling global MI signal Phi(t)
  - MI-weighted graph construction and network metrics
  - Transfer entropy asymmetry index A_i
"""

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Result conventions (do not remove — every notebook calls these)
# ---------------------------------------------------------------------------

def format_result(key, value, comment=None):
    """Print a greppable [RESULT] line and return the formatted string."""
    line = f"[RESULT] {key} = {value}"
    if comment:
        line += f"  # {comment}"
    print(line)
    return line


def save_results(data: dict, path):
    """Write a results dict as JSON. Use /kaggle/working/<phase>_results.json."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {p}")


# ---------------------------------------------------------------------------
# Kraskov k-NN mutual information estimator
# ---------------------------------------------------------------------------

def kraskov_mi(x, y, k=5):
    """Estimate I(X; Y) using the Kraskov k-NN estimator (Algorithm 1).

    Parameters
    ----------
    x, y : array-like, shape (n_samples,)
        1-D continuous signals (equal length).
    k : int
        Number of nearest neighbors (default 5).

    Returns
    -------
    mi : float
        Estimated mutual information in nats (log base e).
    """
    from scipy.special import digamma
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = len(x)

    # Joint space with Chebyshev (max) metric
    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # Distance to k-th neighbor in joint space (Chebyshev)
    dists, _ = tree_xy.query(xy, k=k + 1, workers=-1)
    eps = dists[:, -1]  # k-th neighbor distance, shape (n,)

    # Vectorized batched query — pass all points and per-point radii at once.
    # Scipy 1.7+ supports array-valued r in query_ball_point.
    # return_length=True avoids materialising the list-of-lists, ~3x faster.
    nx = tree_x.query_ball_point(x, eps, p=np.inf, return_length=True) - 1
    ny = tree_y.query_ball_point(y, eps, p=np.inf, return_length=True) - 1

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(max(mi, 0.0))  # MI is non-negative; numerical noise can push below 0


# ---------------------------------------------------------------------------
# Rolling global MI signal Phi(t)
# ---------------------------------------------------------------------------

def rolling_global_mi(data, window_samples, step_samples, k=5, max_channels=64):
    """Compute the global mean pairwise MI signal Phi(t).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Preprocessed iEEG recording.
    window_samples : int
        Samples per rolling window (e.g. 5 s * 500 Hz = 2500).
    step_samples : int
        Step between windows (e.g. 1 s * 500 Hz = 500).
    k : int
        k-NN neighbors for Kraskov MI.
    max_channels : int
        If n_channels > max_channels, select top-variance channels.

    Returns
    -------
    phi : ndarray, shape (n_windows,)
        Global mean pairwise MI at each window center.
    window_centers : ndarray, shape (n_windows,)
        Sample index of each window center.
    """
    n_channels, n_samples = data.shape

    # Select channels by variance if needed
    if n_channels > max_channels:
        variances = np.var(data, axis=1)
        idx = np.argsort(variances)[-max_channels:]
        data = data[idx]
        n_channels = max_channels

    starts = np.arange(0, n_samples - window_samples + 1, step_samples)
    phi = np.empty(len(starts))
    window_centers = starts + window_samples // 2

    for w_idx, start in enumerate(starts):
        end = start + window_samples
        win = data[:, start:end]

        mi_vals = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                mi_vals.append(kraskov_mi(win[i], win[j], k=k))

        if mi_vals:
            # Use median + IQR clipping to suppress outlier pairs (bridged
            # electrodes, artifacts) that would inflate the mean.
            arr = np.array(mi_vals)
            q75 = np.percentile(arr, 75)
            # Clip at 3 * 75th-percentile (keeps real high-MI pairs, removes
            # extreme outliers like ground-bridged channel pairs).
            arr = np.clip(arr, 0.0, max(3.0 * q75, 1e-6))
            phi[w_idx] = float(np.mean(arr))
        else:
            phi[w_idx] = 0.0

        if (w_idx + 1) % 10 == 0:
            print(f"  Window {w_idx + 1}/{len(starts)} done", flush=True)

    return phi, window_centers


def baseline_quality_check(phi_inter, min_mean=0.005, min_snr=0.5):
    """Flag an interictal Phi(t) baseline as unreliable.

    Two failure modes observed in HUP data:
      1. Near-zero baseline (μ < 0.005 nats) — flat/artifact interictal recording.
      2. Low SNR (mean/std < 0.5) — MI signal is pure noise.

    Returns (ok: bool, reason: str).
    """
    mu  = float(np.mean(phi_inter))
    std = float(np.std(phi_inter)) + 1e-12
    snr = mu / std
    if mu < min_mean:
        return False, f"near-zero baseline μ={mu:.5f} < {min_mean}"
    if snr < min_snr:
        return False, f"low SNR={snr:.2f} < {min_snr}"
    return True, "ok"


def detect_collapse(phi, baseline_mask, threshold_sigmas=2.0, min_persist_windows=10):
    """Detect the MI collapse time in Phi(t) via threshold crossing.

    Parameters
    ----------
    phi : ndarray
        Global MI signal.
    baseline_mask : boolean ndarray
        True for windows that belong to the interictal baseline.
    threshold_sigmas : float
        Collapse threshold = mean - threshold_sigmas * std of baseline.
    min_persist_windows : int
        Minimum consecutive windows below threshold to declare collapse.

    Returns
    -------
    collapse_idx : int or None
        Index in phi where collapse is detected, or None if not found.
    threshold : float
        The computed threshold value.
    """
    mu = np.mean(phi[baseline_mask])
    sigma = np.std(phi[baseline_mask])
    threshold = mu - threshold_sigmas * sigma

    below = phi < threshold
    count = 0
    for i, b in enumerate(below):
        if b:
            count += 1
            if count >= min_persist_windows:
                return i - min_persist_windows + 1, threshold
        else:
            count = 0
    return None, threshold


def preictal_trend_analysis(phi_prei, phi_inter, step_sec=5):
    """Analyse the preictal Phi(t) signal using multiple complementary methods.

    Because HUP recordings have only ~2 min of preictal data, a hard threshold
    crossing is often underpowered. This function adds:
      1. Mann-Kendall trend test — non-parametric monotonic decrease test
      2. Linear slope of Phi(t) over the preictal window (nats/s)
      3. Cohen's d effect size: (mu_inter - mu_prei) / sigma_inter
      4. Normalised MI drop: (mu_inter - mu_prei) / mu_inter × 100 (%)
      5. Early vs late preictal comparison (first half vs second half)

    Parameters
    ----------
    phi_prei  : ndarray — rolling Phi(t) during preictal epoch
    phi_inter : ndarray — rolling Phi(t) during interictal baseline
    step_sec  : float   — step between windows in seconds

    Returns
    -------
    dict with keys: slope_nats_per_s, mk_tau, mk_pvalue, cohens_d,
                    pct_drop, early_mean, late_mean, early_late_pvalue
    """
    from scipy import stats

    mu_inter  = float(np.mean(phi_inter))
    sig_inter = float(np.std(phi_inter))
    mu_prei   = float(np.mean(phi_prei))
    n = len(phi_prei)

    # 1. Linear slope via OLS
    t = np.arange(n) * step_sec
    if n >= 2:
        slope, intercept, r, p_slope, se = stats.linregress(t, phi_prei)
    else:
        slope, r, p_slope = 0.0, 0.0, 1.0

    # 2. Mann-Kendall trend test (no scipy built-in; manual implementation)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = phi_prei[j] - phi_prei[i]
            if diff > 0:   s += 1
            elif diff < 0: s -= 1
    # Variance of S under H0 (no ties approximation)
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if var_s > 0:
        z_mk = (s - np.sign(s)) / np.sqrt(var_s)
        p_mk = float(2 * (1 - stats.norm.cdf(abs(z_mk))))
        tau_mk = s / (n * (n - 1) / 2)
    else:
        z_mk, p_mk, tau_mk = 0.0, 1.0, 0.0

    # 3. Cohen's d (inter mean minus prei mean, normalised by inter std)
    cohens_d = (mu_inter - mu_prei) / (sig_inter + 1e-12)

    # 4. Percentage drop
    pct_drop = 100.0 * (mu_inter - mu_prei) / (mu_inter + 1e-12)

    # 5. Early vs late preictal halves
    half = max(1, n // 2)
    early = phi_prei[:half]
    late  = phi_prei[half:]
    if len(early) >= 2 and len(late) >= 2:
        _, p_el = stats.mannwhitneyu(early, late, alternative="greater")
        early_mean = float(np.mean(early))
        late_mean  = float(np.mean(late))
    else:
        p_el, early_mean, late_mean = 1.0, mu_prei, mu_prei

    return {
        "slope_nats_per_s":   round(float(slope), 8),
        "r_squared":          round(float(r ** 2), 4),
        "p_slope":            round(float(p_slope), 4),
        "mk_tau":             round(tau_mk, 4),
        "mk_pvalue":          round(p_mk, 4),
        "cohens_d":           round(cohens_d, 4),
        "pct_drop":           round(pct_drop, 3),
        "early_mean":         round(early_mean, 6),
        "late_mean":          round(late_mean, 6),
        "early_late_pvalue":  round(p_el, 4),
        "mu_inter":           round(mu_inter, 6),
        "mu_prei":            round(mu_prei, 6),
    }


# ---------------------------------------------------------------------------
# Graph-theoretic network metrics
# ---------------------------------------------------------------------------

def mi_network_metrics(data, window_samples, step_samples, k=5,
                       edge_percentile=80, max_channels=64,
                       fixed_threshold=None):
    """Compute MI-weighted network metrics over time.

    Returns density, mean_degree, and clustering coefficient for each window.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    window_samples, step_samples, k, max_channels : see rolling_global_mi
    edge_percentile : float
        Retain edges above this percentile of weights (default top 20%).
        Ignored when fixed_threshold is provided.
    fixed_threshold : float or None
        If given, apply this fixed MI value as the edge threshold for all
        windows instead of computing a per-window percentile.  Pass the value
        returned by mi_network_threshold() computed on the interictal data so
        that preictal density is measured against a stable baseline.

    Returns
    -------
    metrics : dict with keys 'density', 'mean_degree', 'clustering'
        Each value is an ndarray of shape (n_windows,).
    window_centers : ndarray
    """
    n_channels, n_samples = data.shape

    if n_channels > max_channels:
        variances = np.var(data, axis=1)
        idx = np.argsort(variances)[-max_channels:]
        data = data[idx]
        n_channels = max_channels

    starts = np.arange(0, n_samples - window_samples + 1, step_samples)
    n_windows = len(starts)
    window_centers = starts + window_samples // 2

    density = np.empty(n_windows)
    mean_degree = np.empty(n_windows)
    clustering = np.empty(n_windows)

    for w_idx, start in enumerate(starts):
        end = start + window_samples
        win = data[:, start:end]

        # Build full weight matrix
        W = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                mi = kraskov_mi(win[i], win[j], k=k)
                W[i, j] = mi
                W[j, i] = mi

        # Apply threshold: fixed baseline threshold or per-window percentile
        if fixed_threshold is not None:
            threshold = fixed_threshold
        else:
            all_weights = W[np.triu_indices(n_channels, k=1)]
            threshold = np.percentile(all_weights, edge_percentile)
        A = (W >= threshold).astype(float)
        np.fill_diagonal(A, 0)

        n_edges = A.sum() / 2
        max_edges = n_channels * (n_channels - 1) / 2
        density[w_idx] = n_edges / max_edges

        deg = A.sum(axis=1)
        mean_degree[w_idx] = deg.mean()

        # Clustering coefficient (Watts-Strogatz)
        clust = np.zeros(n_channels)
        for i in range(n_channels):
            neighbors = np.where(A[i] > 0)[0]
            ki = len(neighbors)
            if ki < 2:
                clust[i] = 0.0
                continue
            subgraph = A[np.ix_(neighbors, neighbors)]
            triangles = subgraph.sum() / 2
            clust[i] = triangles / (ki * (ki - 1) / 2)
        clustering[w_idx] = clust.mean()

    return {"density": density, "mean_degree": mean_degree,
            "clustering": clustering}, window_centers


def mi_network_threshold(data, window_samples, step_samples, k=5,
                         edge_percentile=80, max_channels=64):
    """Compute a fixed MI edge threshold from a baseline (interictal) epoch.

    Pools all pairwise MI values across all windows and returns the
    edge_percentile-th quantile.  Pass the result as fixed_threshold to
    mi_network_metrics so that preictal density is measured against a stable
    interictal baseline rather than a per-window threshold.
    """
    n_channels, n_samples = data.shape
    if n_channels > max_channels:
        idx = np.argsort(np.var(data, axis=1))[-max_channels:]
        data = data[idx]
        n_channels = max_channels

    starts = np.arange(0, n_samples - window_samples + 1, step_samples)
    all_weights = []
    for start in starts:
        win = data[:, start:start + window_samples]
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                all_weights.append(kraskov_mi(win[i], win[j], k=k))

    return float(np.percentile(all_weights, edge_percentile)) if all_weights else 0.0


# ---------------------------------------------------------------------------
# Transfer Entropy asymmetry index A_i
# ---------------------------------------------------------------------------

def transfer_entropy(source, target, lag=10, embed_dim=3):
    """Estimate TE_{source -> target} using conditional MI.

    Simplified Schreiber estimator via histogram-based conditional entropy.
    For production, replace with IDTxl's Kraskov-based TE.

    Parameters
    ----------
    source, target : array-like, shape (n_samples,)
    lag : int
        Embedding lag in samples.
    embed_dim : int
        Embedding dimension.

    Returns
    -------
    te : float
        Transfer entropy estimate in nats.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(target)

    # Build delay-embedded vectors
    max_lag = lag * embed_dim
    if n <= max_lag + 1:
        return 0.0

    # Target future: t+1
    y_future = target[max_lag + 1:]
    # Target past: k-dimensional embedding
    y_past = np.column_stack([target[max_lag - lag * d: n - 1 - lag * d]
                               for d in range(embed_dim)])
    # Source past: l-dimensional embedding
    x_past = np.column_stack([source[max_lag - lag * d: n - 1 - lag * d]
                               for d in range(embed_dim)])

    # Discretize for histogram-based estimation (8 bins)
    bins = 8
    y_f_d = np.digitize(y_future, np.linspace(y_future.min(), y_future.max(), bins))

    def entropy_discrete(x):
        _, counts = np.unique(x, return_counts=True, axis=0)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-12))

    # TE = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    # Approximated as MI gain using joint discretization
    y_past_d = np.apply_along_axis(
        lambda col: np.digitize(col, np.linspace(col.min(), col.max(), bins)),
        0, y_past)
    x_past_d = np.apply_along_axis(
        lambda col: np.digitize(col, np.linspace(col.min(), col.max(), bins)),
        0, x_past)

    joint_yx = np.column_stack([y_f_d.reshape(-1, 1), y_past_d])
    joint_yxx = np.column_stack([y_f_d.reshape(-1, 1), y_past_d, x_past_d])

    h_y_given_ypast = entropy_discrete(joint_yx) - entropy_discrete(y_past_d)
    h_y_given_ypast_xpast = entropy_discrete(joint_yxx) - entropy_discrete(
        np.column_stack([y_past_d, x_past_d]))

    te = h_y_given_ypast - h_y_given_ypast_xpast
    return float(max(te, 0.0))


def te_asymmetry_index(data, lag=10, embed_dim=3):
    """Compute TE asymmetry index A_i = TE_{i->rest} - TE_{rest->i} for all channels.

    The 'information black hole' hypothesis: SOZ channels have negative A_i
    (more incoming TE than outgoing) at seizure onset.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        iEEG segment covering the ictal onset window.
    lag, embed_dim : TE parameters.

    Returns
    -------
    A : ndarray, shape (n_channels,)
        Asymmetry index per channel. Negative = information sink (SOZ candidate).
    te_matrix : ndarray, shape (n_channels, n_channels)
        Full TE matrix where te_matrix[i, j] = TE_{i->j}.
    """
    n_channels = data.shape[0]
    te_matrix = np.zeros((n_channels, n_channels))

    total_pairs = n_channels * (n_channels - 1)
    done = 0
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            te_matrix[i, j] = transfer_entropy(data[i], data[j], lag=lag,
                                                embed_dim=embed_dim)
            done += 1
            if done % 20 == 0:
                print(f"  TE pairs: {done}/{total_pairs}", flush=True)

    te_out = te_matrix.sum(axis=1) / (n_channels - 1)   # mean outgoing TE
    te_in = te_matrix.sum(axis=0) / (n_channels - 1)    # mean incoming TE
    A = te_out - te_in
    return A, te_matrix


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def bandpass_filter(data, lowcut=0.5, highcut=300.0, fs=500.0, order=4):
    """Apply zero-phase Butterworth bandpass filter."""
    from scipy.signal import butter, sosfiltfilt
    nyq = fs / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch_filter(data, freq=60.0, fs=500.0, Q=30.0):
    """Apply IIR notch filter (and harmonics up to Nyquist)."""
    from scipy.signal import iirnotch, sosfilt
    filtered = data.copy()
    harmonic = freq
    while harmonic < fs / 2:
        b, a = iirnotch(harmonic / (fs / 2), Q)
        from scipy.signal import tf2sos
        sos = tf2sos(b, a)
        filtered = sosfilt(sos, filtered, axis=-1)
        harmonic += freq
    return filtered


def common_average_reference(data):
    """Subtract the mean across all channels (CAR)."""
    return data - data.mean(axis=0, keepdims=True)
