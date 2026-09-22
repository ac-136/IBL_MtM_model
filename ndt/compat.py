"""Load checkpoints produced by the original, un-vendored tree.

`Trainer.save_model` pickles the live `nn.Module`, not a `state_dict`:

    torch.save({"model": self.model, "epoch": epoch}, ...)

Pickle records each object's class by import path, so a checkpoint written by
the original repository names `models.ndt1.NDT1` and `models.masker.Masker`.
Those top-level modules do not exist once the code is vendored under `ndt`, so
`torch.load` raises `ModuleNotFoundError: No module named 'models'`.

The classes themselves are unchanged, so registering module aliases is enough.

    from ndt.compat import load_checkpoint
    ckpt = load_checkpoint("/path/to/model_best.pt")
    model = ckpt["model"]

Prefer re-saving as a plain `state_dict` once loaded. A pickled module is
fragile in ways an alias cannot fix: it also binds the checkpoint to the class
definition, so any later edit to `NDT1.__init__` can break old files.
"""

import sys

# Legacy top-level name -> module in this package.
_ALIASES = {
    "models": "ndt.models",
    "models.ndt1": "ndt.models.ndt1",
    "models.masker": "ndt.models.masker",
    "models.model_output": "ndt.models.model_output",
    "utils": "ndt.utils",
    "utils.config_utils": "ndt.utils.config_utils",
    "loader": "ndt.loader",
    "loader.base": "ndt.loader.base",
    "trainer": "ndt.trainer",
    "trainer.base": "ndt.trainer.base",
}


def install_legacy_aliases(force=False):
    """Make legacy module paths resolve into this package.

    Existing entries in sys.modules are left alone unless force=True, so this is
    safe to call inside the original repo where a real `models` package exists.
    Returns the aliases that were installed.
    """
    import importlib
    installed = []
    for legacy, target in _ALIASES.items():
        if legacy in sys.modules and not force:
            continue
        try:
            sys.modules[legacy] = importlib.import_module(target)
            installed.append(legacy)
        except ImportError:
            pass
    return installed


def load_checkpoint(path, map_location="cpu", **kwargs):
    """torch.load a checkpoint from either tree, old or new."""
    import torch
    install_legacy_aliases()
    kwargs.setdefault("weights_only", False)
    return torch.load(path, map_location=map_location, **kwargs)


def load_state_dict(path, map_location="cpu"):
    """Just the weights, whether the file holds a module or a state_dict."""
    obj = load_checkpoint(path, map_location=map_location)
    model = obj.get("model", obj) if isinstance(obj, dict) else obj
    return model.state_dict() if hasattr(model, "state_dict") else model


def resave_as_state_dict(src_path, dst_path, map_location="cpu"):
    """Convert a pickled-module checkpoint into a portable state_dict file.

    The result depends on no local module paths and survives edits to the model
    class, so this is the durable fix for the 92 legacy checkpoints.
    """
    import torch
    obj = load_checkpoint(src_path, map_location=map_location)
    out = {"state_dict": load_state_dict(src_path, map_location=map_location)}
    if isinstance(obj, dict) and "epoch" in obj:
        out["epoch"] = obj["epoch"]
    torch.save(out, dst_path)
    return dst_path
