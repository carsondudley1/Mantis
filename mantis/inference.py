import torch
from .model import MultiTimeSeriesForecaster
from .utils import preprocess_input

class Mantis:
    def __init__(self, model_path: str, forecast_horizon: int = 8, use_covariate: bool = True):
        """
        Args:
            model_path: Path to the .pt file (e.g., 'models/mantis_8w_cov.pt')
            forecast_horizon: Number of weeks to forecast (4 or 8)
            use_covariate: Whether the model expects a covariate input
        """
        self.use_covariate = use_covariate
        self.forecast_horizon = forecast_horizon

        # Define model config (should match training)
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
            teacher_forcing_ratio=0.0,
            dropout=0.2
        )

        # Load weights
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, time_series, covariate=None, population=None, target_type=2):
        """
        Run Mantis on a single time series.

        Args:
            time_series: 1D array-like of daily target values (raw)
            covariate: 1D array-like of daily covariate values (raw), or None
            population: Optional float population (will be log1p transformed)
            target_type: 0=cases, 1=hosp, 2=deaths

        Returns:
            np.ndarray of shape [H, 9] with predicted quantiles
        """
        # Sanity check
        if self.use_covariate and covariate is None:
            raise ValueError("This model expects a covariate input, but none was provided.")

        # Get input dict for the model
        inputs = preprocess_input(
            time_series,
            covariate if self.use_covariate else None,
            population,
            target_type=target_type
        )

        with torch.no_grad():
            pred = self.model.predict(**inputs)  # [1, H, 9]

        pred = pred.squeeze(0).cpu().numpy()  # [H, 9]

        # Widen all quantiles away from the median (index 4)
        median = pred[:, 4:5]  # shape [H, 1]
        widened = median + 1.15 * (pred - median)
        widened[:, 4] = median[:, 0]  # don't shift the median itself

        return widened
