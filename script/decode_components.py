"""Behaviour-decoding components: targets, heads, losses, metrics. No I/O.

The frozen-NDT1 probe (``behavior_decode.py``) and the raw-spike baseline
(``decode_baseline.py``) both import this module, so their target
preprocessing is identical by construction.

Component-level parity with NEDS (reference copy: ``neds_scripts/``):

* targets and encodings -- choice -1/+1 -> 0/1 and block 0.2/0.5/0.8 -> 0/1/2
  (NEDS ``base.py:449-453``); wheel speed and whisker motion energy as 100-bin
  traces, min-max scaled with NEDS's formula (``ibl_data_utils.py:686-689``);
* head shapes -- static: a learned weight over the T tokens (init
  ``torch.rand(T)``, ``mm.py:120-122``) sum-pools (B, T, H) to (B, H), then a
  linear map to the classes; dynamic: a per-token linear map to one value
  (``mm.py:253-277``);
* loss types -- cross-entropy for static targets, MSE for dynamic ones
  (``mm.py:92-95``); the total is the unweighted sum (``mm.py:233``);
* metrics -- accuracy and balanced accuracy; PSTH R2 over (block, choice,
  reward) conditions and single-trial R2 over every (trial, bin) entry,
  both unclipped (``eval_utils.py:757-775``). ``subtract_psth`` in NEDS only
  changes plots, so nothing is subtracted here either.

Deliberate departures (see results/claude/decoding.md):

* every statistic that shapes preprocessing comes from the TRAIN split only:
  min-max bounds, the >=30%-invalid drop rule, class presence, and the
  chance/null estimates. NEDS scales over all trials of all splits;
* invalid entries are masked per target instead of deleting the trial, so the
  MtM split membership is kept;
* loss reduction is an explicit masked mean per component (CE over valid
  trials, MSE over valid (trial, bin) entries). NEDS divides the static CE by
  the time-expanded mask count (B x T), an implicit 1/T weight not carried
  over.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score

STATIC = ("choice", "block")
DYNAMIC = ("wheel-speed", "whisker-motion-energy")
TARGETS = STATIC + DYNAMIC
SPLITS = ("train", "val", "test")
N_CLASSES = {"choice": 2, "block": 3}
# NEDS base.py:449-450, keyed on the float value rather than its string.
LOOKUP = {"choice": {-1.0: 0, 1.0: 1}, "block": {0.2: 0, 0.5: 1, 0.8: 2}}
# NEDS drops a behaviour when >= 30% of its trials are NaN (ibl_data_utils.py:657).
MAX_INVALID_FRAC = 0.3
# NEDS plot_psth conditions on these trial variables (eval_utils.py:264-270).
PSTH_CONDITIONS = ("block", "choice", "reward")


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

def encode_static(name, raw):
    """Raw per-trial values -> (int64 labels, valid mask). Unmapped values
    (choice == 0 no-go trials, NaN) are masked; their label is 0 and unused."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    labels = np.zeros(len(raw), dtype=np.int64)
    mask = np.zeros(len(raw), dtype=bool)
    for value, label in LOOKUP[name].items():
        # 1e-6 so float32-stored labels (0.2 -> 0.20000000298) map as NEDS maps
        # them via str(np.float32(x)); the label values are >= 0.3 apart.
        hit = np.isclose(raw, value, rtol=0, atol=1e-6)
        labels[hit] = label
        mask |= hit
    return labels, mask


def minmax_scale(x, lo, hi):
    """NEDS's formula (x - min) / (max - min), with externally supplied bounds."""
    return (x - lo) / (hi - lo)


def prepare_targets(raw):
    """Build decoding targets with strict split isolation.

    ``raw[split][name]``: static names (n,) raw values; dynamic names (n, T)
    raw traces with NaN for invalid entries. Every decision is made on train:
    the drop rule, the min-max bounds, class presence. Val and test only
    receive the train-fitted transform (no clipping).

    Returns ``{"targets": {split: {name: {"y", "mask"}}}, "info": {name: {...}}}``.
    ``mask`` is (n,) for static targets and (n, T) for dynamic ones.
    """
    targets = {split: {} for split in SPLITS}
    info = {}

    for name in STATIC:
        enc = {split: encode_static(name, raw[split][name]) for split in SPLITS}
        train_labels, train_mask = enc["train"]
        invalid_frac = 1.0 - float(train_mask.mean()) if len(train_mask) else 1.0
        present = sorted(set(train_labels[train_mask].tolist()))
        dropped = None
        if invalid_frac >= MAX_INVALID_FRAC:
            dropped = f"train invalid fraction {invalid_frac:.3f} >= {MAX_INVALID_FRAC}"
        elif len(present) < 2:
            dropped = f"train has {len(present)} class(es)"
        info[name] = {
            "kind": "static",
            "dropped": dropped,
            "train_invalid_frac": invalid_frac,
            "train_classes_present": present,
            "n_masked": {split: int((~enc[split][1]).sum()) for split in SPLITS},
            "n": {split: int(len(enc[split][1])) for split in SPLITS},
        }
        for split in SPLITS:
            labels, mask = enc[split]
            targets[split][name] = {
                "y": labels,
                "mask": mask & (dropped is None),
            }

    for name in DYNAMIC:
        values = {split: np.asarray(raw[split][name], dtype=np.float64) for split in SPLITS}
        train = values["train"]
        train_finite = np.isfinite(train)
        trial_invalid = ~train_finite.any(axis=1)
        invalid_frac = float(trial_invalid.mean()) if len(trial_invalid) else 1.0
        lo = float(train[train_finite].min()) if train_finite.any() else float("nan")
        hi = float(train[train_finite].max()) if train_finite.any() else float("nan")
        dropped = None
        if invalid_frac >= MAX_INVALID_FRAC:
            dropped = f"train invalid fraction {invalid_frac:.3f} >= {MAX_INVALID_FRAC}"
        elif not hi > lo:
            dropped = f"train max {hi} <= min {lo}"
        info[name] = {
            "kind": "dynamic",
            "dropped": dropped,
            "train_invalid_frac": invalid_frac,
            "train_min": lo,
            "train_max": hi,
            "n_invalid_trials": {
                split: int((~np.isfinite(values[split]).any(axis=1)).sum()) for split in SPLITS
            },
            "n_nan_entries": {split: int(np.isnan(values[split]).sum()) for split in SPLITS},
            "n": {split: int(len(values[split])) for split in SPLITS},
        }
        for split in SPLITS:
            x = values[split]
            mask = np.isfinite(x) & (dropped is None)
            y = minmax_scale(x, lo, hi) if dropped is None else np.zeros_like(x)
            targets[split][name] = {
                "y": np.where(mask, y, 0.0),
                "mask": mask,
            }

    return {"targets": targets, "info": info}


def active_targets(prepared):
    return [n for n in TARGETS if prepared["info"][n]["dropped"] is None]


# --------------------------------------------------------------------------
# Heads and losses
# --------------------------------------------------------------------------

class StaticHead(nn.Module):
    """NEDS static head: token weights (init U[0,1)) sum-pool, then linear."""

    def __init__(self, n_tokens, hidden, n_classes):
        super().__init__()
        self.token_weight = nn.Parameter(torch.rand(n_tokens))
        self.proj = nn.Linear(hidden, n_classes)

    def forward(self, x):                       # (B, T, H) -> (B, C)
        pooled = (x * self.token_weight[None, :, None]).sum(1)
        return self.proj(pooled)


class DynamicHead(nn.Module):
    """NEDS dynamic head: one linear map per token to a scalar."""

    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x):                       # (B, T, H) -> (B, T)
        return self.proj(x).squeeze(-1)


def build_heads(names, n_tokens, hidden):
    """One head per active target. Built in TARGETS order so RNG use is fixed."""
    heads = nn.ModuleDict()
    for name in TARGETS:
        if name not in names:
            continue
        if name in STATIC:
            heads[name] = StaticHead(n_tokens, hidden, N_CLASSES[name])
        else:
            heads[name] = DynamicHead(hidden)
    return heads


def component_loss(name, pred, y, mask):
    """Masked mean for one target. Returns (loss, n_valid).

    static:  sum(CE_i * m_i) / sum(m_i)          over trials
    dynamic: sum((p - y)^2 * m) / sum(m)         over (trial, bin) entries
    An all-masked batch contributes an exact 0 (no gradient) with n_valid 0.
    """
    m = mask.to(pred.dtype)
    n_valid = mask.sum()
    # Neutralise masked targets BEFORE the loss: a NaN target would otherwise
    # give NaN gradients through torch.where below (0 * NaN).
    y = torch.where(mask, y, torch.zeros_like(y))
    if name in STATIC:
        per = nn.functional.cross_entropy(pred, y, reduction="none")
    else:
        per = (pred - y) ** 2
    per = torch.where(mask, per, torch.zeros_like(per))
    if n_valid == 0:
        return per.sum() * 0.0, 0
    return (per * m).sum() / m.sum(), int(n_valid)


def total_loss(heads, x, batch_targets):
    """Unweighted sum of the per-component masked means, plus each component."""
    components = {}
    total = x.new_zeros(())
    for name, head in heads.items():
        t = batch_targets[name]
        loss, n_valid = component_loss(name, head(x), t["y"], t["mask"])
        components[name] = (loss, n_valid)
        total = total + loss
    return total, components


# --------------------------------------------------------------------------
# Metrics (NEDS eval_utils.py:463-508 / plot_psth, via ndt's identical helpers)
# --------------------------------------------------------------------------

def _ndt_r2_helpers():
    from ndt.utils.eval_utils import compute_all_psth, compute_R2_psth, compute_R2_main
    return compute_all_psth, compute_R2_psth, compute_R2_main


def static_metrics(y, pred):
    return {"acc": float(accuracy_score(y, pred)),
            "balanced_acc": float(balanced_accuracy_score(y, pred))}


def dynamic_metrics(y, pred, conditions):
    """PSTH R2 and single-trial R2 exactly as NEDS's plot_psth computes them.

    y, pred: (K, T) on complete trials; conditions: (K, 3) raw
    (block, choice, reward) labels of those trials, used only for grouping.
    """
    compute_all_psth, compute_R2_psth, compute_R2_main = _ndt_r2_helpers()
    X = np.repeat(conditions[:, None, :], y.shape[1], axis=1)   # (K, T, 3)
    idxs = np.arange(conditions.shape[1])
    r2_psth = compute_R2_psth(compute_all_psth(X, y, idxs),
                              compute_all_psth(X, pred, idxs), clip=False)
    r2_trial = compute_R2_main(y.reshape(-1, 1), pred.reshape(-1, 1), clip=False)[0]
    return {"r2_psth": float(r2_psth), "r2_trial": float(r2_trial)}


def evaluable(name, target):
    """Rows used for a metric: valid labels (static) or trials with every bin
    finite (dynamic; NEDS's R2 has no notion of a partially valid trial)."""
    mask = target["mask"]
    return mask if name in STATIC else mask.all(axis=1)


def metrics_for(name, target, pred, conditions):
    rows = evaluable(name, target)
    out = {"n_eval": int(rows.sum()), "n_excluded": int((~rows).sum())}
    if rows.sum() == 0:
        return out
    if name in STATIC:
        out.update(static_metrics(target["y"][rows], np.asarray(pred)[rows]))
    else:
        out.update(dynamic_metrics(target["y"][rows], np.asarray(pred)[rows], conditions[rows]))
    return out


def bootstrap_ci(name, target, pred, conditions, n_boot=1000, seed=0):
    """95% percentile CI per metric from resampling evaluable test trials."""
    rows = np.flatnonzero(evaluable(name, target))
    if len(rows) < 2:
        return {}
    rng = np.random.default_rng(seed)
    y, p, c = target["y"][rows], np.asarray(pred)[rows], conditions[rows]
    draws = {}
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), len(rows))
        if name in STATIC:
            m = static_metrics(y[idx], p[idx])
        else:
            m = dynamic_metrics(y[idx], p[idx], c[idx])
        for k, v in m.items():
            draws.setdefault(k, []).append(v)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in draws.items()}


# --------------------------------------------------------------------------
# Chance / null (train only)
# --------------------------------------------------------------------------

def chance_predictions(name, train_target, n_test):
    """Train-only null: majority class (static) or per-bin train mean (dynamic)."""
    if name in STATIC:
        labels = train_target["y"][train_target["mask"]]
        majority = int(np.bincount(labels, minlength=N_CLASSES[name]).argmax())
        return np.full(n_test, majority, dtype=np.int64), {"majority_class": majority}
    y, m = train_target["y"], train_target["mask"]
    mean = np.where(m.any(0), (y * m).sum(0) / np.maximum(m.sum(0), 1), 0.0)
    return np.tile(mean, (n_test, 1)), {"train_mean_trace": mean.tolist()}
