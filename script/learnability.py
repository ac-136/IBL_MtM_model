"""Own-session learnability: the band rule, its bootstrap, and the diagnostics.

Pure numpy, no torch, so the math can be tested offline and shared by the eval
script and the report. Nothing here chooses a threshold from data: the band is
defined against the mean-rate null that ``bits_per_spike`` already uses.
"""

import numpy as np

LN2 = np.log(2.0)
B_DEFAULT = 1000
SEED_DEFAULT = 0

BAND_RULE = (
    "Band = own-session co-smoothing bits/spike (mean over finite neurons) vs the "
    "per-neuron mean-rate null. 95% paired percentile bootstrap over TEST TRIALS "
    "(B=1000, seed 0, same resampled trials for every neuron), with the null held at "
    "the full-sample test mean. CI lower > 0 -> '>0'; CI upper < 0 -> '<0'; "
    "CI contains 0 (within 1e-9 numerical tolerance) -> '~0'. Fixed before any "
    "own-session value was seen."
)


def per_trial_terms(pred, gt):
    """Per-trial pieces of ``bits_per_spike`` for ONE neuron.

    ``pred``/``gt`` are the arrays ``co_smoothing_eval`` hands to
    ``bits_per_spike``: shape ``[trials, T]`` or ``[trials, T, 1]``.

    ``neg_log_likelihood`` sums ``r - y*log r + lgamma(y+1)`` over cells. The
    lgamma term is identical for model and null and cancels in the difference,
    so it is omitted. The null rate is the neuron's mean over ALL test cells,
    exactly as ``bits_per_spike`` fits it.

    Returns ``(s, d, lam)``: per-trial spike count, per-trial
    (null NLL - model NLL), and the fixed null rate.
    """
    p = np.asarray(pred, dtype=np.float64).reshape(pred.shape[0], -1)
    y = np.asarray(gt, dtype=np.float64).reshape(gt.shape[0], -1)
    if np.isnan(y).any():
        raise ValueError("NaN in ground-truth spikes; per-trial decomposition "
                         "would not match bits_per_spike's NaN masking")
    p = np.where(p == 0, 1e-9, p)            # mirrors neg_log_likelihood
    lam = y.mean()
    s = y.sum(axis=1)
    model = (p - y * np.log(p)).sum(axis=1)
    if lam > 0:
        null = lam * y.shape[1] - s * np.log(lam)
    else:
        null = np.full_like(s, np.nan)      # no spikes at all: bps undefined
    return s, null - model, lam


def bps_from_terms(s, d):
    """Bits per spike from per-trial terms. Accepts [trials] or [..., trials]."""
    S = np.sum(s, axis=-1)
    D = np.sum(d, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = D / S / LN2
    return np.where(np.isfinite(out), out, np.nan)


def bootstrap_session_mean(s, d, B=B_DEFAULT, seed=SEED_DEFAULT):
    """Paired bootstrap of the session-mean bps over test trials.

    ``s`` and ``d`` are ``[trials, neurons]``. Each resample draws trials with
    replacement -- the SAME draw for every neuron, preserving between-neuron
    correlation -- and recomputes each neuron's bps with the null held at its
    full-sample value, then averages over finite neurons.

    Holding the null fixed is deliberate. Refitting it per resample would hand
    the null an in-sample advantage the model does not get: a model predicting
    exactly the test mean scores exactly 0 on the full data but strictly < 0 on
    every refit resample, and would be misclassified '<0' instead of '~0'.

    Returns ``(lo, hi, draws)`` with a 95% percentile interval.
    """
    s = np.asarray(s, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    n = s.shape[0]
    rng = np.random.default_rng(seed)
    # Multinomial counts instead of index arrays: [B, trials] @ [trials, N] keeps
    # memory at O(B*N) rather than O(B*trials*N).
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
    d_ok = np.where(np.isfinite(d), d, 0.0)
    S = W @ s
    D = W @ d_ok
    dead = ~np.all(np.isfinite(d), axis=0)            # neurons with undefined null
    with np.errstate(divide="ignore", invalid="ignore"):
        per = D / S / LN2
    per[:, dead] = np.nan
    per[~np.isfinite(per)] = np.nan
    draws = np.nanmean(per, axis=1)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi), draws


# Numerical tolerance, NOT an effect-size threshold. A model that predicts the
# test mean exactly has a bootstrap CI of about [-1e-16, -1e-16] from float64
# rounding alone; a strict comparison with 0 would call that '<0'. 1e-9
# bits/spike is ~1e6x above float64 rounding here and orders of magnitude
# below any effect a neural model could have, so it separates arithmetic noise
# from signal without encoding a judgement about what counts as "learned".
NUMERICAL_EPS = 1e-9


def band(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    if lo > NUMERICAL_EPS:
        return ">0"
    if hi < -NUMERICAL_EPS:
        return "<0"
    return "~0"


def variation_ratios(pred, gt):
    """How much the PREDICTION varies, relative to the data, for one neuron.

    temporal: var over time of the trial-averaged prediction / same for spikes.
    trial:    var over trials of the time-averaged prediction / same for spikes.
    ~0 means the prediction is flat along that axis -- the signature of a
    mean-like solution. NaN where the data itself has no variance.
    """
    p = np.asarray(pred, dtype=np.float64).reshape(pred.shape[0], -1)
    y = np.asarray(gt, dtype=np.float64).reshape(gt.shape[0], -1)
    vt_y, vr_y = y.mean(0).var(), y.mean(1).var()
    vt = p.mean(0).var() / vt_y if vt_y > 0 else np.nan
    vr = p.mean(1).var() / vr_y if vr_y > 0 else np.nan
    return float(vt), float(vr)
