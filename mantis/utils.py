import numpy as np
import torch

# Hardcoded normalization stats (match training values exactly)
DEFAULT_MEAN = {
    0: 5.55,  # cases
    1: 3.84,  # hospitalizations
    2: 2.57   # deaths
}
DEFAULT_STD = {
    0: 3.63,
    1: 3.15,
    2: 2.59
}

def preprocess_input(
    time_series,
    covariate=None,
    population=None,
    target_type=2,
    mean_std=None  # optionally pass in custom normalization dict
):
    """
    Converts raw daily time series (and covariate) into model-ready input.
    
    Args:
        time_series: 1D array-like of daily target values (raw)
        covariate: 1D array-like of daily covariate values (raw), or None
        population: float (or None). Will be log1p-transformed and normalized.
        target_type: 0=cases, 1=hosp, 2=deaths
        mean_std: Optional dict with 'mean' and 'std' overrides
    
    Returns:
        Dict suitable to pass into model.predict(...)
    """
    # Convert to float32 np arrays
    x = np.array(time_series, dtype=np.float32)
    cov = np.array(covariate, dtype=np.float32) if covariate is not None else None

    # Normalize with global stats
    mean = mean_std['mean'] if mean_std else DEFAULT_MEAN[target_type]
    std  = mean_std['std'] if mean_std else DEFAULT_STD[target_type]

    x = (np.log1p(x) - mean) / (std + 1e-7)

    if cov is not None:
        cov_mean = mean_std['cov_mean'] if mean_std else DEFAULT_MEAN[target_type]
        cov_std  = mean_std['cov_std'] if mean_std else DEFAULT_STD[target_type]
        cov = (np.log1p(cov) - cov_mean) / (cov_std + 1e-7)
        feats = np.stack([x, cov], axis=1)
    else:
        feats = np.stack([x, np.zeros_like(x)], axis=1)  # dummy 0 covariate

    values = torch.tensor(feats).unsqueeze(0)  # [1, T, 2]
    valid_mask = torch.ones(values.shape[1], dtype=torch.bool).unsqueeze(0)  # [1, T]
    day_indices = torch.arange(values.shape[1]).unsqueeze(0) + 1098

    pop_log = np.log1p(population) if population is not None else 0.0
    population_tensor = torch.tensor([pop_log], dtype=torch.float32)

    return {
        'values': values,
        'disease_type': torch.tensor([0], dtype=torch.long),  # dummy for now
        'target_type': torch.tensor([target_type], dtype=torch.long),
        'population': population_tensor,
        'day_indices': day_indices,
        'valid_mask': valid_mask
    }
