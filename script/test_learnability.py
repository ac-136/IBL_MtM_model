#!/usr/bin/env python
"""Offline checks for script/learnability.py. No GPU, no data, seconds.

    python script/test_learnability.py

Each check pins a property the band rule depends on, against ndt's own
``bits_per_spike`` where one exists, so the decomposition is tested against the
production function rather than against a restatement of itself.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script"))

from ndt.utils.eval_utils import bits_per_spike  # noqa: E402
from learnability import (per_trial_terms, bps_from_terms,  # noqa: E402
                          bootstrap_session_mean, band, variation_ratios)

results = []


def check(name, ok, detail=""):
    results.append((bool(ok), name, detail))


rng = np.random.default_rng(123)
TR, T, N = 120, 100, 40
# A time-varying ground-truth rate per neuron, and Poisson spikes from it.
t = np.linspace(0, 1, T)
true = 0.05 + 0.15 * np.exp(-((t[None, :] - rng.uniform(0.2, 0.8, (N, 1))) ** 2) / 0.01)
true = np.broadcast_to(true.T[None], (TR, T, N)).copy()      # [trials, T, N]
gt = rng.poisson(true).astype(np.float64)


def session(pred):
    ss, dd, bp = [], [], []
    for n in range(N):
        s, d, _ = per_trial_terms(pred[:, :, [n]], gt[:, :, [n]])
        ss.append(s); dd.append(d)
        bp.append(bps_from_terms(s, d))
    return np.stack(ss, 1), np.stack(dd, 1), np.array(bp)


# 1. the decomposition reproduces ndt's bits_per_spike (float64 inputs)
pred_noisy = true * rng.uniform(0.7, 1.3, true.shape)
_, _, mine = session(pred_noisy)
ref = np.array([bits_per_spike(pred_noisy[:, :, [n]].copy(), gt[:, :, [n]].copy())
                for n in range(N)])
err = np.nanmax(np.abs(mine - ref))
check("1. per-trial decomposition == ndt bits_per_spike (float64)",
      err < 1e-9, f"max |diff| = {err:.2e} over {N} neurons")

# 2. predicting exactly the per-neuron test mean gives exactly 0
mean_pred = np.broadcast_to(gt.mean(axis=(0, 1), keepdims=True), gt.shape).copy()
s2, d2, b2 = session(mean_pred)
check("2. constant predictor at the test mean -> bps exactly 0",
      np.nanmax(np.abs(b2)) < 1e-9, f"max |bps| = {np.nanmax(np.abs(b2)):.2e}")
lo, hi, _ = bootstrap_session_mean(s2, d2)
check("2b. ...and its band is '~0' (the refit-null bias is NOT present)",
      band(lo, hi) == "~0", f"CI [{lo:+.2e}, {hi:+.2e}] -> {band(lo, hi)}")

# 2c. non-vacuity for the fixed-null design: REFITTING the null per resample
# really does bias a mean-predictor below 0, by far more than rounding. If this
# failed, holding the null fixed would be an unmotivated choice.
Wn = np.random.default_rng(0).multinomial(TR, np.full(TR, 1 / TR), size=1000)
refit = []
for w in Wn:
    idx = np.repeat(np.arange(TR), w)
    vals = [bits_per_spike(mean_pred[idx][:, :, [n]].copy(), gt[idx][:, :, [n]].copy())
            for n in range(N)]
    refit.append(np.nanmean(vals))
rhi = np.percentile(refit, 97.5)
check("2c. refit-null bootstrap would misclassify the mean-predictor as '<0'",
      rhi < -1e-6, f"refit CI upper = {rhi:+.2e}  (bias the fixed-null design removes)")

# 3. a constant predictor at the wrong rate is worse than the null
off_pred = mean_pred * 1.6
s3, d3, b3 = session(off_pred)
lo, hi, _ = bootstrap_session_mean(s3, d3)
check("3. constant predictor at 1.6x the mean -> '<0'",
      band(lo, hi) == "<0" and np.nanmean(b3) < 0,
      f"mean {np.nanmean(b3):+.4f}, CI [{lo:+.4f}, {hi:+.4f}] -> {band(lo, hi)}")

# 4. the true time-varying rate beats the null, with a CI excluding 0
s4, d4, b4 = session(true)
lo, hi, _ = bootstrap_session_mean(s4, d4)
check("4. true time-varying rate -> '>0'",
      band(lo, hi) == ">0", f"mean {np.nanmean(b4):+.4f}, CI [{lo:+.4f}, {hi:+.4f}]")

# 5. the variation ratios separate a flat prediction from a structured one
vt_flat = np.nanmedian([variation_ratios(mean_pred[:, :, n], gt[:, :, n])[0] for n in range(N)])
vt_true = np.nanmedian([variation_ratios(true[:, :, n], gt[:, :, n])[0] for n in range(N)])
check("5. temporal-variation ratio: flat ~0, true rate clearly > 0",
      vt_flat < 1e-12 and vt_true > 0.1, f"flat {vt_flat:.2e}, true {vt_true:.3f}")

# 6. a neuron with no test spikes is NaN everywhere, never silently 0
gt_dead = gt.copy(); gt_dead[:, :, 0] = 0
s6, d6, _ = per_trial_terms(true[:, :, [0]], gt_dead[:, :, [0]])
check("6. zero-spike neuron -> bps NaN (as ndt: inf/0 -> nan)",
      np.isnan(bps_from_terms(s6, d6)), f"bps = {bps_from_terms(s6, d6)}")

# 7. the bootstrap is deterministic under its fixed seed
a = bootstrap_session_mean(s4, d4)[:2]; b = bootstrap_session_mean(s4, d4)[:2]
check("7. bootstrap deterministic under seed 0", a == b, f"{a} == {b}")

print("=" * 72)
for ok, name, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
print("=" * 72)
npass = sum(ok for ok, _, _ in results)
print(f"{npass}/{len(results)} passed")
sys.exit(0 if npass == len(results) else 1)
