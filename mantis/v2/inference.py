from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
from chronos.chronos2 import Chronos2Model


class Mantis2:
    """Generate 16-week probabilistic forecasts from a Mantis 2 checkpoint."""

    context_length = 512
    prediction_length = 16

    def __init__(self, checkpoint_dir="models/mantis2", device="cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        for filename in ("config.json", "model.safetensors"):
            if not (checkpoint_dir / filename).is_file():
                raise FileNotFoundError(checkpoint_dir / filename)

        self.device = torch.device(device)
        self.model = Chronos2Model.from_pretrained(
            str(checkpoint_dir), local_files_only=True
        ).to(self.device).eval()
        self.quantiles = tuple(self.model.chronos_config.quantiles)

    @torch.inference_mode()
    def predict(self, time_series):
        """Forecast raw weekly values as (targets, quantiles, forecast weeks).

        Supply one series as a 1-D array or up to six aligned series as
        rows of a 2-D array, ordered oldest to newest. Use NaN or negative
        values for missing observations. Seasonal features are generated
        from a fixed synthetic start date.
        """
        history = np.asarray(time_series, dtype=np.float32)
        if history.ndim == 1:
            history = history[None, :]
        if history.ndim != 2:
            raise ValueError("time_series must have shape (weeks,) or (targets, weeks).")

        n_targets, n_weeks = history.shape
        if not 1 <= n_targets <= 6 or n_weeks < 8:
            raise ValueError("Provide 1 to 6 targets with at least 8 weeks of history.")

        # Keep the seasonal phase anchored to the first supplied week,
        # even when only the most recent 512 weeks are used.
        start = max(0, n_weeks - self.context_length)
        history_weeks = n_weeks - start
        first_week = date(2000, 1, 3)
        iso_weeks = np.array([
            (first_week + timedelta(weeks=offset)).isocalendar()[1]
            for offset in range(start, n_weeks + self.prediction_length)
        ])
        angle = 2 * np.pi * iso_weeks / 52.18
        calendar = np.stack([np.sin(angle), np.cos(angle)]).astype(np.float32)

        history = history[:, start:]
        targets = np.arcsinh(np.where(history < 0, np.nan, history))
        context = np.concatenate([targets, calendar[:, :history_weeks]], axis=0)
        future_covariates = np.concatenate(
            [
                np.full((n_targets, self.prediction_length), np.nan, dtype=np.float32),
                calendar[:, history_weeks:],
            ],
            axis=0,
        )

        output = self.model(
            context=torch.from_numpy(context).to(self.device),
            future_covariates=torch.from_numpy(future_covariates).to(self.device),
            group_ids=torch.zeros(n_targets + 2, dtype=torch.long, device=self.device),
            num_output_patches=1,
        )
        return torch.sinh(output.quantile_preds[:n_targets]).float().cpu().numpy()
