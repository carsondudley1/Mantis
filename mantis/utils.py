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
    covariate_type=None,
    mean_std=None  # optional dict: {'mean','std','cov_mean','cov_std','pop_mean','pop_std'}
):
    """
    Converts raw WEEKLY time series (and optional covariate) into model-ready input.

    Args:
        time_series: 1D array-like of weekly target values (raw)
        covariate: 1D array-like of weekly covariate values (raw), or None
        population: float (or None). log1p-transformed; standardized if pop stats provided.
        target_type: 0=cases, 1=hosp, 2=deaths
        covariate_type: 0=cases, 1=hosp, 2=deaths (optional; defaults to target_type)
        mean_std: Optional dict with overrides:
                  {'mean','std','cov_mean','cov_std','pop_mean','pop_std'}

    Returns:
        Dict suitable to pass into model.predict(...)
    """
    # Convert to float32 np arrays
    x = np.asarray(time_series, dtype=np.float32)
    cov = np.asarray(covariate, dtype=np.float32) if covariate is not None else None

    # Use passed-in stats or defaults (for TARGET)
    if mean_std is not None and 'mean' in mean_std and 'std' in mean_std:
        target_mean = float(mean_std['mean'])
        target_std  = float(mean_std['std'])
    else:
        target_mean = DEFAULT_MEAN[target_type]
        target_std  = DEFAULT_STD[target_type]

    x = (np.log1p(x) - target_mean) / (target_std + 1e-7)

    # Use correct stats for COVARIATE (if provided)
    if cov is not None:
        cov_type = covariate_type if covariate_type is not None else target_type
        if mean_std is not None and 'cov_mean' in mean_std and 'cov_std' in mean_std:
            cov_mean = float(mean_std['cov_mean'])
            cov_std  = float(mean_std['cov_std'])
        else:
            cov_mean = DEFAULT_MEAN[cov_type]
            cov_std  = DEFAULT_STD[cov_type]

        cov = (np.log1p(cov) - cov_mean) / (cov_std + 1e-7)
        feats = np.stack([x, cov], axis=1)              # [T, 2]
    else:
        # TRUE 1-channel path when no covariate
        feats = x[:, None]                              # [T, 1]

    values = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # [1, T, C]
    T = values.shape[1]
    valid_mask = torch.ones(T, dtype=torch.bool).unsqueeze(0)       # [1, T]
    day_indices = torch.arange(T, dtype=torch.long).unsqueeze(0) + 1098

    # Population: log1p, then standardize if pop stats provided
    pop_log = float(np.log1p(float(population))) if population is not None else 0.0
    if mean_std is not None and 'pop_mean' in mean_std and 'pop_std' in mean_std:
        population_scaled = (pop_log - float(mean_std['pop_mean'])) / (float(mean_std['pop_std']) + 1e-7)
    else:
        population_scaled = pop_log  # fallback if no pop stats provided

    population_tensor = torch.tensor([population_scaled], dtype=torch.float32)

    return {
        'values': values,
        'disease_type': torch.tensor([0], dtype=torch.long),  # dummy for now
        'target_type': torch.tensor([target_type], dtype=torch.long),
        'population': population_tensor,
        'day_indices': day_indices,
        'valid_mask': valid_mask
    }
