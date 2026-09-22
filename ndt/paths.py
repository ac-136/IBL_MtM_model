"""Package-relative resource paths.

Every config and data file shipped with this package is resolved against the
package directory, never the process working directory. This is what lets the
package be imported and run from anywhere.
"""

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PKG_DIR / "configs"
DATA_DIR = PKG_DIR / "data"


def config_path(name: str) -> str:
    """Absolute path to a shipped YAML config.

    Accepts either a bare name ("ndt1_stitching") or one with the extension
    and/or a subdirectory ("ppwang/trainer_sl_choice_ndt1.yaml").
    """
    if not name.endswith(".yaml"):
        name = name + ".yaml"
    return str(CONFIG_DIR / name)


def data_path(name: str) -> str:
    """Absolute path to a shipped data asset, e.g. "target_eids.txt"."""
    return str(DATA_DIR / name)
