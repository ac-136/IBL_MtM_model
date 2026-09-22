"""NDT1 for spiking neural data: model, loader, trainer, and evaluation.

Vendored from the IBL_MtM_model repository. Self-contained: all configs and data
assets are resolved relative to this package, never the working directory.

This module deliberately has no imports, so ``import ndt`` does not pull in
torch or read any data file. That matters because ``ndt.models.ndt1`` opens
``data/target_eids.txt`` at module import. Import the pieces directly:

    from ndt.models.ndt1 import NDT1
    from ndt.loader.make_loader import make_loader
    from ndt.trainer.make import make_trainer
    from ndt.utils.dataset_utils import load_ibl_dataset_locally
    from ndt.paths import config_path, data_path

To load a checkpoint written by the original, un-vendored tree, use
``ndt.compat``. A pickled module records its class's import path, so those files
name ``models.ndt1.NDT1``, which does not exist here.
"""

__version__ = "0.2.0"
