from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config_utils import DictConfig


""" Temporal masking for NDT: whole timesteps are masked across all channels.

The MtM masking schemes were removed: ``neuron``, ``random``, ``co-smooth``,
``forward-pred``, ``inter-region`` and ``intra-region``, along with the
``mask_regions`` / ``target_regions`` / ``n_mask_regions`` config they needed.
Held-out neurons and timesteps at evaluation time are produced by
``eval_utils.heldout_mask``, which is independent of this module.

CONFIG:
    force_active: mask even outside training mode
    mode: must be ``temporal``
    ratio: fraction of timesteps to predict; 0 disables masking, which is how
           the ``*_eval.yaml`` configs turn it off at evaluation time
    zero_ratio: of the selected bins, fraction zeroed out
    random_ratio: of the not-zeroed, fraction randomly replaced
    expand_prob: probability of expanding the mask for consecutive bins
    max_timespan: max length of the expanded mask
"""
class Masker(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.force_active = config.force_active if "force_active" in config else False
        self.mode = config.mode
        self.ratio = config.ratio
        self.zero_ratio = config.zero_ratio
        self.random_ratio = config.random_ratio
        self.expand_prob = config.expand_prob
        self.max_timespan = config.max_timespan

    def forward(
        self,
        spikes: torch.FloatTensor,                      # (bs, seq_len, n_channels)
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:    # (bs, seq_len, n_channels) x2

        if not self.training and not self.force_active:
            return spikes, torch.zeros_like(spikes)
        elif self.ratio == 0:
            # how the *_eval.yaml configs disable masking during evaluation
            return spikes, torch.zeros_like(spikes)

        if self.mode != "temporal":
            raise ValueError(
                f"Only temporal masking is supported, got {self.mode!r}. The other "
                "modes were MtM-specific and have been removed."
            )

        mask_ratio = self.ratio

        # Expand mask over consecutive bins. expand_prob is 0 in every shipped
        # config, but the draw is kept so the RNG stream is unchanged.
        if torch.bernoulli(torch.tensor(self.expand_prob).float()):
            timespan = torch.randint(1, self.max_timespan + 1, (1, )).item()
        else:
            timespan = 1
        mask_ratio = mask_ratio / timespan
        mask_probs = torch.full(spikes[:, :, 0].shape, mask_ratio)  # (bs, seq_len)

        # Create mask
        mask = torch.bernoulli(mask_probs).to(spikes.device)

        # Expand mask
        if timespan > 1:
            mask = self.expand_timesteps(mask, timespan)
        mask = mask.unsqueeze(2).expand_as(spikes).bool()

        # Mask data
        zero_idx = torch.bernoulli(torch.full(spikes.shape, self.zero_ratio)).to(spikes.device).bool() & mask
        spikes[zero_idx] = 0
        random_idx = torch.bernoulli(torch.full(spikes.shape, self.random_ratio)).to(spikes.device).bool() & mask & ~zero_idx
        random_spikes = (spikes.max() * torch.rand(spikes.shape, device=spikes.device)).to(spikes.dtype)
        spikes[random_idx] = random_spikes[random_idx]

        return spikes, mask.to(torch.int64)

    @staticmethod
    def expand_timesteps(mask, width=1):
        kernel = torch.ones(width, device=mask.device).view(1, 1, -1)
        expanded_mask = F.conv1d(mask.unsqueeze(1), kernel, padding="same")
        return (expanded_mask.squeeze(1) >= 1)
