import math

import torch


def build_lr_scheduler(optimizer, config, steps_per_epoch):
    scheduler_name = str(getattr(config.optimizer, "scheduler", "cosine")).lower()
    gradient_accumulation_steps = max(
        1, int(getattr(config.optimizer, "gradient_accumulation_steps", 1))
    )
    total_steps = max(
        1,
        config.training.num_epochs * steps_per_epoch // gradient_accumulation_steps,
    )
    warmup_pct = float(getattr(config.optimizer, "warmup_pct", 0.0) or 0.0)
    warmup_steps = min(total_steps - 1, int(total_steps * warmup_pct))

    if scheduler_name in {"none", "constant"}:
        return None

    if scheduler_name == "cosine":
        div_factor = float(getattr(config.optimizer, "div_factor", 10.0) or 10.0)
        start_scale = 1.0 / max(div_factor, 1.0)

        def lr_lambda(current_step):
            if warmup_steps > 0 and current_step < warmup_steps:
                progress = current_step / max(1, warmup_steps)
                return start_scale + (1.0 - start_scale) * progress

            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    elif scheduler_name == "linear":

        def lr_lambda(current_step):
            if warmup_steps > 0 and current_step < warmup_steps:
                return current_step / max(1, warmup_steps)

            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return max(0.0, 1.0 - progress)

    elif scheduler_name == "step":
        gamma = float(getattr(config.optimizer, "gamma", 0.95) or 0.95)

        def lr_lambda(current_step):
            if warmup_steps > 0 and current_step < warmup_steps:
                return current_step / max(1, warmup_steps)

            post_warmup_step = max(0, current_step - warmup_steps)
            epoch_idx = post_warmup_step // max(1, steps_per_epoch)
            return gamma ** epoch_idx

    else:
        raise ValueError(
            f"Unsupported optimizer scheduler '{scheduler_name}'. "
            "Expected one of: cosine, linear, step, none."
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
