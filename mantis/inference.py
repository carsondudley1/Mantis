import os
import torch
import numpy as np
from .model import MultiTimeSeriesForecaster
from .utils import preprocess_input, DEFAULT_MEAN, DEFAULT_STD

class Mantis:
    def __init__(self, forecast_horizon: int = 4, use_covariate: bool = True, model_dir: str = "models"):
        """
        Args:
            forecast_horizon: Number of weeks to forecast (4 or 8)
            use_covariate: Whether the model uses a covariate input
            model_dir: Path to directory containing the model .pt files
        """
        assert forecast_horizon in [4, 8], "forecast_horizon must be 4 or 8"
        self.use_covariate = use_covariate
        self.forecast_horizon = forecast_horizon

        suffix = "cov" if use_covariate else "nocov"
        filename = f"mantis_{forecast_horizon}w_{suffix}.pt"
        model_path = os.path.join(model_dir, filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize model with known architecture
        self.model = MultiTimeSeriesForecaster(
            input_window=112,
            forecast_horizon=forecast_horizon,
            hidden_dim=1024,
            ffn_dim=2048,
            n_layers=16,
            n_heads=16,
            n_quantiles=9,
            disease_embed_dim=64,
            pop_embed_dim=64,
            binary_feat_dim=32,
            dropout=0.2
        )

        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, time_series, covariate=None, population=None, target_type=2, covariate_type=None):
        """
        Run Mantis on a single time series.

        Args:
            time_series: 1D array-like of daily target values (raw)
            covariate: 1D array-like of daily covariate values (raw), or None
            population: Optional float population (log1p-transformed internally)
            target_type: 0=cases, 1=hosp, 2=deaths
            covariate_type: 0=cases, 1=hosp, 2=deaths (optional; defaults to target_type)

        Returns:
            np.ndarray of shape [H, 9] with predicted quantiles (denormalized)
        """
        if self.use_covariate and covariate is None:
            raise ValueError("This model expects a covariate input, but none was provided.")

        inputs = preprocess_input(
            time_series,
            covariate if self.use_covariate else None,
            population,
            target_type=target_type,
            covariate_type=covariate_type if self.use_covariate else None
        )

        with torch.no_grad():
            pred = self.model.predict(**inputs)  # [1, H, 9]

        pred = pred.squeeze(0).cpu().numpy()  # [H, 9]

        # Widen all quantiles away from the median (except the median itself)
        median = pred[:, 4:5]
        widened = median + 1.15 * (pred - median)
        widened[:, 4] = median[:, 0]  # Preserve exact median

        # Denormalize: reverse z-score and expm1
        mean = DEFAULT_MEAN[target_type]
        std = DEFAULT_STD[target_type]
        denorm = np.expm1(widened * std + mean)

        return denorm
