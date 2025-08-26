import os
import torch
import numpy as np
from .model import MultiTimeSeriesForecaster
from .utils import preprocess_input, DEFAULT_MEAN, DEFAULT_STD


def _strip_module_prefix(state_dict):
    """Handle DataParallel checkpoints by stripping 'module.' prefix if present."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _remap_ts_type_keys(state_dict):
    """
    Training used ts_type_* while the inference model uses target_type_*.
    Remap older checkpoint keys to current names (no-op if already current).
    """
    mapping = {
        "ts_type_embedding.weight": "target_type_embedding.weight",
        "ts_type_norm.weight":      "target_type_norm.weight",
        "ts_type_norm.bias":        "target_type_norm.bias",
    }
    for old, new in mapping.items():
        if old in state_dict and new not in state_dict:
            state_dict[new] = state_dict.pop(old)
    return state_dict


def _detect_input_channels(state_dict):
    """
    Detect whether checkpoint expects 1 or 2 input channels from the first conv.
    Returns 1 (nocov) or 2 (cov).
    """
    probe_keys = [
        "values_embedding.conv_short.weight",
        "values_embedding.conv_med.weight",
        "values_embedding.conv_long.weight",
        "values_embedding.conv_vlong.weight",
    ]
    for k in probe_keys:
        if k in state_dict:
            return int(state_dict[k].shape[1])
    # Fallback: assume nocov if unknown
    return 1


class Mantis:
    def __init__(self, forecast_horizon: int = 4, use_covariate: bool = True, model_dir: str = "models"):
        """
        Args:
            forecast_horizon: Number of weeks to forecast (4 or 8)
            use_covariate: Whether you intend to use a covariate input
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

        # --- load checkpoint early so we can configure the model correctly ---
        state_dict = torch.load(model_path, map_location="cpu")
        state_dict = _strip_module_prefix(state_dict)
        state_dict = _remap_ts_type_keys(state_dict)

        # Determine checkpoint’s expected input channels (1 = nocov, 2 = cov)
        ckpt_in_ch = _detect_input_channels(state_dict)

        # Sanity-check against the user's intention
        if self.use_covariate and ckpt_in_ch == 1:
            raise ValueError(
                "You requested use_covariate=True but the checkpoint expects 1 input channel (nocov). "
                "Load a *_cov.pt checkpoint or set use_covariate=False."
            )
        if (not self.use_covariate) and ckpt_in_ch == 2:
            raise ValueError(
                "You requested use_covariate=False but the checkpoint expects 2 input channels (cov). "
                "Load a *_nocov.pt checkpoint or set use_covariate=True and provide a covariate."
            )

        # Initialize model with correct input channel count
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
            dropout=0.2,
            # >>> this is the important bit <<<
            values_input_dim=ckpt_in_ch,
        )

        # Load weights strictly now that shapes/names match
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def predict(self, time_series, covariate=None, population=None, target_type=2, covariate_type=None):
        """
        Run Mantis on a single time series.

        Args:
            time_series: 1D array-like of WEEKLY target values (raw)
            covariate:  1D array-like of WEEKLY covariate values (raw), or None
            population: Optional float population (log1p-transformed and optionally standardized)
            target_type: 0=cases, 1=hosp, 2=deaths
            covariate_type: 0=cases, 1=hosp, 2=deaths (optional; defaults to target_type)

        Returns:
            np.ndarray of shape [H, 9] with predicted quantiles (denormalized)
        """
        if self.use_covariate and covariate is None:
            raise ValueError("This model expects a covariate input, but none was provided.")

        inputs = preprocess_input(
            time_series=time_series,
            covariate=covariate if self.use_covariate else None,
            population=population,
            target_type=target_type,
            covariate_type=covariate_type if self.use_covariate else None,
            # Optionally pass pop stats here if you want exact training-style standardization:
            # mean_std={'pop_mean': 14.1607, 'pop_std': 1.9670}
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
