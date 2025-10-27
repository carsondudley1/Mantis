'''

Example training script (could be run on Google colab if connected to GPU)

'''

import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import traceback
import time

# Try to import Google Drive mount functionality
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

################################################################################
## PART 1: DATA LOADING AND PREPROCESSING
################################################################################

def aggregate_to_weekly(daily_data: np.ndarray) -> np.ndarray:
    """
    Aggregate daily data to weekly by summing every 7 days.

    Args:
        daily_data: Array of daily values

    Returns:
        Array of weekly aggregated values
    """
    # Handle potential incomplete last week
    n_days = len(daily_data)
    n_complete_weeks = n_days // 7

    # Reshape to (n_complete_weeks, 7) and sum across dimension 1
    weekly_data = np.sum(daily_data[:n_complete_weeks * 7].reshape(n_complete_weeks, 7), axis=1)

    # Handle remaining days if not divisible by 7
    remaining_days = n_days % 7
    if remaining_days > 0:
        last_week = np.sum(daily_data[n_complete_weeks * 7:])
        weekly_data = np.append(weekly_data, last_week)

    return weekly_data

def pad_collate_fn(batch):
    """
    Pads each sample in the batch to the same sequence length (left-padding).
    Works with weekly data where
        values   ∈ ℝ^{seq_len × 2}   (col-0 = target , col-1 = covariate)
        day_indices ∈ ℤ^{seq_len}    (actual week numbers)
    After padding:
      • values        → [B, L_max, 2]
      • day_indices   → [B, L_max]   (shifted by +1098)
      • valid_mask    → [B, L_max]   (True = real, False = pad)
    """
    SHIFT_VALUE = 1098
    max_len     = max(item['values'].size(0) for item in batch)
    n_feat      = batch[0]['values'].size(1)            # 2
    vals, masks, weeks = [], [], []

    for item in batch:
        seq_len = item['values'].size(0)
        pad_len = max_len - seq_len

        # ---- valid mask ----------------------------------------------------
        mask = torch.cat([
            torch.zeros(pad_len, dtype=torch.bool),
            torch.ones (seq_len, dtype=torch.bool)
        ])
        masks.append(mask.unsqueeze(0))

        # ---- values (left-pad with zeros) ----------------------------------
        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, n_feat, dtype=item['values'].dtype)
            padded_vals = torch.cat([pad_tensor, item['values']], dim=0)
        else:
            padded_vals = item['values']
        vals.append(padded_vals.unsqueeze(0))

        # ---- week indices --------------------------------------------------
        if pad_len > 0:
            first_idx   = item['day_indices'][0].item()
            pad_indices = torch.arange(first_idx - pad_len, first_idx, dtype=torch.long)
            padded_ix   = torch.cat([pad_indices, item['day_indices']])
        else:
            padded_ix   = item['day_indices']
        weeks.append((padded_ix + SHIFT_VALUE).unsqueeze(0))

    batch_values      = torch.cat(vals,   dim=0)   # [B, L_max, 2]
    batch_valid_mask  = torch.cat(masks,  dim=0)   # [B, L_max]
    batch_day_indices = torch.cat(weeks,  dim=0)   # [B, L_max]

    # Static / horizon-length fields
    target_values = torch.stack([b['target_values'] for b in batch])
    population    = torch.stack([b['population']     for b in batch])
    disease_type  = torch.stack([b['disease_type']   for b in batch])
    target_type   = torch.stack([b['target_type']    for b in batch])  # New: target type

    return {
        'values'       : batch_values,
        'day_indices'  : batch_day_indices,
        'valid_mask'   : batch_valid_mask,
        'target_values': target_values,
        'population'   : population,
        'disease_type' : disease_type,
        'target_type'  : target_type,      # New: target type
    }


class GeneralTimeSeriesDatasetWeekly(Dataset):
    """
    General dataset for time series forecasting with any target/covariate combination.
    Y = TARGET_TYPE ; X = past TARGET_TYPE + COVARIATE_TYPE

    Args:
        df: DataFrame with time series data
        global_stats: Dictionary with normalization stats for each time series type
        target_type: Type of target variable (0=cases, 1=hosp, 2=death)
        covariate_type: Type of covariate variable (0=cases, 1=hosp, 2=death)
        max_context: Maximum context window length in weeks
        min_weeks_to_start: Minimum weeks before starting forecasts
        fh: Forecast horizon in weeks
    """
    def __init__(self, df, global_stats, target_type, covariate_type,
                 max_context=112, min_weeks_to_start=8, fh=4):
        super().__init__()
        self.fh = fh
        self.max_context = max_context
        self.min_weeks_to_start = min_weeks_to_start
        self.target_type = target_type
        self.covariate_type = covariate_type

        # Map types to column prefixes and stats keys
        type_to_prefix = {0: 'cases_day_', 1: 'hosp_day_', 2: 'death_day_'}
        type_to_stats = {
            0: ('cases_mean', 'cases_std'),
            1: ('hosp_mean', 'hosp_std'),
            2: ('death_mean', 'death_std')
        }

        # Get appropriate normalization stats
        target_mean_key, target_std_key = type_to_stats[target_type]
        cov_mean_key, cov_std_key = type_to_stats[covariate_type]

        self.target_mean = global_stats[target_mean_key]
        self.target_std = global_stats[target_std_key]
        self.cov_mean = global_stats[cov_mean_key]
        self.cov_std = global_stats[cov_std_key]

        # Get column names for target and covariate
        target_prefix = type_to_prefix[target_type]
        cov_prefix = type_to_prefix[covariate_type]

        target_cols = sorted([c for c in df if c.startswith(target_prefix)],
                           key=lambda c: int(c.split('_')[-1]))
        cov_cols = sorted([c for c in df if c.startswith(cov_prefix)],
                         key=lambda c: int(c.split('_')[-1]))

        if not target_cols:
            raise ValueError(f"No columns found for target type {target_type} (prefix: {target_prefix})")
        if not cov_cols:
            raise ValueError(f"No columns found for covariate type {covariate_type} (prefix: {cov_prefix})")

        # Convert daily data to weekly for each row
        self.target_weekly = [aggregate_to_weekly(r[target_cols].to_numpy('float32'))
                             for _, r in df.iterrows()]
        self.cov_weekly = [aggregate_to_weekly(r[cov_cols].to_numpy('float32'))
                          for _, r in df.iterrows()]

        # Generate all valid forecast start indices
        self.valid = []
        for row in range(len(df)):
            max_weeks = len(self.target_weekly[row])
            max_end = max_weeks - fh
            for end in range(min_weeks_to_start, max_end + 1):
                self.valid.append((row, end))

        # Static features (disease_type, population)
        self.disease_types = torch.zeros(len(df), dtype=torch.long)

        # Population normalization (same as original)
        pop = torch.log1p(torch.tensor(df['population'].values, dtype=torch.float32))
        self.populations = (pop - pop.mean()) / (pop.std() + 1e-7)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        row, end = self.valid[idx]
        ctx_len = min(self.max_context, end)
        start = end - ctx_len

        # Get context data (input features)
        target_ctx = self.target_weekly[row][start:end]      # target context
        cov_ctx = self.cov_weekly[row][start:end]           # covariate context

        # Get future target data (what we're trying to predict)
        target_future = self.target_weekly[row][end:end + self.fh]

        # Apply log1p transformation and normalization
        # Target context uses target stats, covariate context uses covariate stats
        target_scaled = (np.log1p(target_ctx) - self.target_mean) / self.target_std
        cov_scaled = (np.log1p(cov_ctx) - self.cov_mean) / self.cov_std

        # Future target uses target stats
        target_future_scaled = (np.log1p(target_future) - self.target_mean) / self.target_std

        # Stack target and covariate as input features [seq_len, 2]
        values = torch.from_numpy(np.stack([target_scaled, cov_scaled], 1).astype('float32'))

        # Week indices for temporal embeddings
        week_indices = torch.arange(start, end, dtype=torch.long)

        return dict(
            values=values,
            target_values=torch.from_numpy(target_future_scaled.astype('float32')),
            population=self.populations[row],
            disease_type=self.disease_types[row],
            target_type=torch.tensor(self.target_type, dtype=torch.long),  # New: target type
            day_indices=week_indices
        )


def compute_global_stats_by_time_series_type(data_dir):
    """
    Reads all CSVs from data_dir, aggregates to weekly,
    logs them, and computes global_mean, global_std separately for
    cases, hospitalizations, and deaths.

    Returns:
        dict: Dictionary with separate stats for each time series type
    """
    csv_files = glob.glob(os.path.join(data_dir, "disease_dataset_*.csv"))

    # Initialize storage for all types
    all_weekly_values = {
        'cases': [],
        'hosp': [],
        'death': []
    }

    for file in csv_files:
        df = pd.read_csv(file)

        # Get columns by type
        case_cols = [c for c in df.columns if c.startswith('cases_day_')]
        hosp_cols = [c for c in df.columns if c.startswith('hosp_day_')]
        death_cols = [c for c in df.columns if c.startswith('death_day_')]

        # Process each row in the dataframe
        for _, row in df.iterrows():
            # Process cases
            if case_cols:
                daily_values = row[case_cols].values.astype('float32')
                weekly_values = aggregate_to_weekly(daily_values)
                all_weekly_values['cases'].extend(weekly_values)

            # Process hospitalizations
            if hosp_cols:
                daily_values = row[hosp_cols].values.astype('float32')
                weekly_values = aggregate_to_weekly(daily_values)
                all_weekly_values['hosp'].extend(weekly_values)

            # Process deaths
            if death_cols:
                daily_values = row[death_cols].values.astype('float32')
                weekly_values = aggregate_to_weekly(daily_values)
                all_weekly_values['death'].extend(weekly_values)

    # Compute stats for each type
    stats = {}

    # Cases
    cases_weekly = np.array(all_weekly_values['cases'])
    cases_log = np.log1p(cases_weekly)
    stats['cases_mean'] = cases_log.mean()
    stats['cases_std'] = cases_log.std() + 1e-7

    # Hospitalizations
    hosp_weekly = np.array(all_weekly_values['hosp'])
    hosp_log = np.log1p(hosp_weekly)
    stats['hosp_mean'] = hosp_log.mean()
    stats['hosp_std'] = hosp_log.std() + 1e-7

    # Deaths
    death_weekly = np.array(all_weekly_values['death'])
    death_log = np.log1p(death_weekly)
    stats['death_mean'] = death_log.mean()
    stats['death_std'] = death_log.std() + 1e-7

    return stats

def create_multitarget_dataloaders(
    data_dir,
    global_stats,
    target_type=None,        # 0=cases, 1=hosp, 2=death, or None for all combinations
    covariate_type=None,     # 0=cases, 1=hosp, 2=death, or None for all combinations
    max_context_weeks=16,
    min_weeks_to_start=4,
    forecast_horizon=4,
    batch_size=32,
    num_workers=4
):
    """
    Creates DataLoaders for disease forecasting.

    Args:
        data_dir: Directory containing disease dataset CSV files
        global_stats: Dictionary with normalization stats for each time series type
        target_type: Target time series type (0=cases, 1=hosp, 2=death).
                    If None, creates datasets for all valid combinations.
        covariate_type: Covariate time series type (0=cases, 1=hosp, 2=death).
                       If None, creates datasets for all valid combinations.
        max_context_weeks: Maximum context window length
        min_weeks_to_start: Minimum weeks before starting forecasts
        forecast_horizon: Number of weeks to forecast
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Find all disease dataset files
    csv_files = glob.glob(os.path.join(data_dir, "disease_dataset_*.csv"))
    if not csv_files:
        raise ValueError(f"No disease dataset files found in {data_dir}")

    # Determine which target/covariate combinations to create
    if target_type is not None and covariate_type is not None:
        # Single specific combination
        if target_type == covariate_type:
            raise ValueError("target_type and covariate_type must be different")
        combinations = [(target_type, covariate_type)]
        print(f"Creating datasets for single combination: target_type={target_type}, covariate_type={covariate_type}")
    else:
        # All valid combinations (target != covariate)
        combinations = [
            (0, 1),  # cases from cases + hosp
            (0, 2),  # cases from cases + death
            (1, 0),  # hosp from hosp + cases
            (1, 2),  # hosp from hosp + death
            (2, 0),  # death from death + cases
            (2, 1),  # death from death + hosp
        ]
        print(f"Creating datasets for all {len(combinations)} valid combinations")

    # Create datasets for each combination
    all_datasets = []
    total_examples = 0

    for target_t, cov_t in combinations:
        combination_datasets = []
        combination_examples = 0

        for file in csv_files:
            df = pd.read_csv(file)
            print(f"Loading {file}: {len(df)} rows for target_type={target_t}, covariate_type={cov_t}")

            # Create dataset for this file and combination
            try:
                dataset = GeneralTimeSeriesDatasetWeekly(
                    df, global_stats, target_t, cov_t,
                    max_context_weeks, min_weeks_to_start, forecast_horizon)
                combination_datasets.append(dataset)
                combination_examples += len(dataset)

            except ValueError as e:
                print(f"Skipping {file} for combination ({target_t}, {cov_t}): {e}")
                continue

        if combination_datasets:
            # Combine datasets for this combination
            combination_dataset = ConcatDataset(combination_datasets)
            all_datasets.append(combination_dataset)
            total_examples += combination_examples

            type_names = {0: "cases", 1: "hosp", 2: "death"}
            print(f"  Combination {type_names[target_t]} from {type_names[target_t]}+{type_names[cov_t]}: {combination_examples} examples")

    if not all_datasets:
        raise ValueError("No valid datasets could be created")

    # Combine all datasets into one
    full_dataset = ConcatDataset(all_datasets)
    print(f"Total examples across all combinations: {total_examples}")

    # Create train dataloader
    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_collate_fn
    )

    # Create validation loader (same as train for compatibility)
    val_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate_fn
    )

    return train_loader, val_loader

def get_population_stats(data_dir):
    """
    Extract population statistics (mean and std of log-transformed population)
    from the training datasets.
    """
    # Get all dataset files
    csv_files = glob.glob(os.path.join(data_dir, "disease_dataset_*.csv"))

    if not csv_files:
        raise ValueError(f"No dataset files found in {data_dir}")

    # Collect all population values
    all_populations = []

    for file in csv_files:
        df = pd.read_csv(file)
        if 'population' not in df.columns:
            continue
        all_populations.extend(df['population'].values)

    if not all_populations:
        raise ValueError("No population data found in datasets")

    # Convert to numpy array
    all_populations = np.array(all_populations)

    # Apply log1p transform
    log_populations = np.log1p(all_populations)

    # Calculate mean and std
    pop_mean = np.mean(log_populations)
    pop_std = np.std(log_populations) + 1e-7  # Add small epsilon to avoid division by zero

    return pop_mean, pop_std

################################################################################
## PART 2: MODEL ARCHITECTURE
################################################################################

class MultiScaleConvEmbedding(nn.Module):
    """
    Multi-scale convolutional embedding layer that captures patterns at different time scales.

    This module uses parallel convolutions with different kernel sizes to capture:
    - Short-term patterns (e.g., 1-3 days)
    - Medium-term patterns (e.g., weekly cycles)
    - Long-term patterns (e.g., incubation periods, seasonal effects)
    """
    def __init__(self, input_dim=1, output_dim=512, dropout=0.1):
        super().__init__()

        # Different kernel sizes to capture multi-scale patterns
        self.conv_short = nn.Conv1d(input_dim, output_dim // 4, kernel_size=3, padding=1)
        self.conv_med = nn.Conv1d(input_dim, output_dim // 4, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(input_dim, output_dim // 4, kernel_size=15, padding=7)
        self.conv_vlong = nn.Conv1d(input_dim, output_dim // 4, kernel_size=31, padding=15)

        # Layer normalization per scale
        self.norm_short = nn.LayerNorm(output_dim // 4)
        self.norm_med = nn.LayerNorm(output_dim // 4)
        self.norm_long = nn.LayerNorm(output_dim // 4)
        self.norm_vlong = nn.LayerNorm(output_dim // 4)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Final projection combining all scales
        self.final_proj = nn.Linear(output_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Embedded tensor of shape [batch_size, seq_len, output_dim]
        """
        # Transpose for 1D convolution [batch_size, input_dim, seq_len]
        x_conv = x.transpose(1, 2)

        # Apply convolutions at different scales
        x_short = self.conv_short(x_conv).transpose(1, 2)  # [B, seq_len, output_dim//4]
        x_med = self.conv_med(x_conv).transpose(1, 2)
        x_long = self.conv_long(x_conv).transpose(1, 2)
        x_vlong = self.conv_vlong(x_conv).transpose(1, 2)

        # Apply normalization and activation
        x_short = self.dropout(self.activation(self.norm_short(x_short)))
        x_med = self.dropout(self.activation(self.norm_med(x_med)))
        x_long = self.dropout(self.activation(self.norm_long(x_long)))
        x_vlong = self.dropout(self.activation(self.norm_vlong(x_vlong)))

        # Concatenate multi-scale features
        x_combined = torch.cat([x_short, x_med, x_long, x_vlong], dim=2)

        # Final projection
        x_out = self.final_proj(x_combined)
        x_out = self.final_norm(x_out)

        return x_out

class TemporalPatternAttention(nn.Module):
    """
    Custom attention mechanism that emphasizes temporal patterns.
    Includes relative positional encoding and adaptive attention span.
    """
    def __init__(self, hidden_dim, n_heads, dropout=0.1, max_len=2000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Relative positional bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_len - 1, n_heads))
        positions = torch.arange(max_len).unsqueeze(1) - torch.arange(max_len).unsqueeze(0)
        positions = positions + max_len - 1  # Shift to [0, 2*max_len-1]
        self.register_buffer('positions', positions)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len]
                  where True indicates valid positions

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention: [batch_size, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add relative positional bias
        rel_pos = self.rel_pos_bias[self.positions[:seq_len, :seq_len]]  # [seq_len, seq_len, n_heads]
        rel_pos = rel_pos.permute(2, 0, 1)  # [n_heads, seq_len, seq_len]
        attn_scores = attn_scores + rel_pos.unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            invalid = ~mask
            invalid = invalid.unsqueeze(1).unsqueeze(2)
            invalid = invalid.expand(-1, self.n_heads, x.size(1), -1)
            attn_scores = attn_scores.masked_fill(invalid, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights
        output = torch.matmul(attn_weights, v)  # [batch_size, n_heads, seq_len, head_dim]

        # Reshape back: [batch_size, seq_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Final projection
        output = self.out_proj(output)

        return output

class CNNTransformerBlock(nn.Module):
    """
    Hybrid block combining CNN for local pattern extraction with transformer for global context.
    """
    def __init__(self, hidden_dim, ffn_dim, n_heads, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        # Local pattern CNN
        self.local_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )

        # Custom pattern-oriented attention
        self.pattern_attn = TemporalPatternAttention(hidden_dim, n_heads, dropout)

        # Layer normalizations
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Local CNN processing - using post-norm architecture
        x_conv = x.transpose(1, 2)
        x_conv = self.local_conv(x_conv).transpose(1, 2)
        x = x + self.dropout(x_conv)
        x = self.norm1(x)

        # Global attention - using post-norm architecture
        x_attn = self.pattern_attn(x, mask)
        x = x + self.dropout(x_attn)
        x = self.norm2(x)

        # Feed-forward network - using post-norm architecture
        x_ffn = self.ffn(x)
        x = x + self.dropout(x_ffn)
        x = self.norm3(x)

        return x

class EpidemicPatternMemory(nn.Module):
    """
    Pattern Memory Bank for disease forecasting that stores learned prototype patterns.

    This module maintains a bank of learned disease pattern prototypes that represent
    common patterns observed across various diseases and outbreaks (e.g., seasonal surges,
    intervention responses, reporting anomalies). The model can match input sequences
    against these patterns and incorporate relevant pattern knowledge into its predictions.
    """
    def __init__(self, hidden_dim, num_patterns=256, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        # Learnable disease pattern prototypes - initialized with small random values
        self.pattern_bank = nn.Parameter(torch.randn(num_patterns, hidden_dim) * 0.02)

        # Pattern matching network - computes similarity between input and patterns
        self.pattern_matcher = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_patterns)
        )

        # Projection for retrieved patterns
        self.pattern_proj = nn.Linear(hidden_dim, hidden_dim)

        # Normalization and regularization
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass through the pattern memory bank.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional boolean mask tensor of shape [batch_size, seq_len]
                  where True indicates valid positions

        Returns:
            Enhanced tensor of shape [batch_size, seq_len, hidden_dim]
        """
        # Apply pre-normalization (similar to transformer blocks)
        x_norm = self.norm1(x)

        # Match input patterns against stored prototypes
        pattern_weights = self.pattern_matcher(x_norm)  # [B, seq_len, num_patterns]

        # Apply attention mask if provided
        if mask is not None:
            # Create mask in correct shape: [B, seq_len, 1]
            mask_expanded = mask.unsqueeze(-1).float()
            # Apply mask (invalid positions will have zero weight)
            pattern_weights = pattern_weights * mask_expanded

        # Apply softmax to get attention distribution over patterns
        pattern_weights = F.softmax(pattern_weights, dim=-1)  # [B, seq_len, num_patterns]

        # Retrieve weighted combination of pattern prototypes
        retrieved_patterns = torch.matmul(pattern_weights, self.pattern_bank)  # [B, seq_len, hidden_dim]

        # Project retrieved patterns
        retrieved_patterns = self.pattern_proj(retrieved_patterns)

        # Apply dropout for regularization
        retrieved_patterns = self.dropout(retrieved_patterns)

        # Combine with input using residual connection
        x_with_patterns = x + retrieved_patterns

        # Apply final layer normalization
        return self.norm2(x_with_patterns)

class MultiTimeSeriesForecaster(nn.Module):
    """
    Enhanced foundation model for disease forecasting using a hybrid CNN-Transformer architecture
    with a pattern memory bank. Now supports multiple time series types (cases, hospitalizations, deaths).

    Key enhancements:
    1. Adds time series type embedding to switch between forecasting different types
    2. Pattern memory bank that stores and recalls common disease patterns
    3. Richer temporal features derived from week indices
    4. Two-layer GRU in the decoder for better sequence modeling
    5. Designed for weekly data with 4-week forecast horizon
    """
    def __init__(
        self,
        input_window=112,  # 112 weeks of context
        forecast_horizon=4,  # 4 weeks forecast
        hidden_dim=512,
        ffn_dim=768,
        n_layers=8,
        n_heads=8,
        n_quantiles=9,
        disease_embed_dim=64,
        pop_embed_dim=64,
        binary_feat_dim=32,
        teacher_forcing_ratio=0.1,
        dropout=0.1,
        layer_norm_eps=1e-5
    ):
        super().__init__()
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_quantiles = n_quantiles
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # ===== FEATURE EMBEDDINGS =====
        # Multi-scale convolutional embedding for time series data
        self.values_embedding = MultiScaleConvEmbedding(
            input_dim=2,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Disease type embedding
        self.disease_embedding = nn.Embedding(3, disease_embed_dim)
        self.disease_norm = nn.LayerNorm(disease_embed_dim, eps=layer_norm_eps)

        # Population embedding
        self.population_mlp = nn.Sequential(
            nn.Linear(1, pop_embed_dim),
            nn.LayerNorm(pop_embed_dim, eps=layer_norm_eps),
            nn.GELU()
        )

        # Enhanced temporal embeddings - derived from day indices
        self.day_of_week_embed = nn.Embedding(7, 128)        # 0-6 for day of week
        self.day_norm = nn.LayerNorm(128, eps=layer_norm_eps)

        # Month embedding - approximate by dividing day index by 30
        self.month_embed = nn.Embedding(12, 128)             # 0-11 for month
        self.month_norm = nn.LayerNorm(128, eps=layer_norm_eps)

        # Day of year embedding - approximate by modulo 365
        self.day_of_year_embed = nn.Embedding(366, 128)      # 0-365 for day of year
        self.day_of_year_norm = nn.LayerNorm(128, eps=layer_norm_eps)

        # Add target type embedding alongside disease type
        self.target_type_embedding = nn.Embedding(3, 128)  # 3 types: cases, hosp, death
        self.target_type_norm = nn.LayerNorm(128, eps=layer_norm_eps)

        # Input feature projection
        input_feat_dim = hidden_dim + disease_embed_dim + pop_embed_dim + 128 + 128 + 128 + 128
        self.feature_projection = nn.Linear(input_feat_dim, hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # ===== ENCODER BLOCKS =====
        self.encoder_blocks = nn.ModuleList([
            CNNTransformerBlock(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                n_heads=n_heads,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(n_layers)
        ])

        # ===== PATTERN MEMORY BANK =====
        # Add pattern memory bank after encoder blocks with 256 patterns
        self.pattern_memory = EpidemicPatternMemory(
            hidden_dim=hidden_dim,
            num_patterns=256,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

        # ===== DECODER COMPONENTS =====
        # Initial decoder state projection
        self.decoder_init_proj = nn.Linear(hidden_dim, hidden_dim)

        # Input processing for each decoder step
        self.decoder_input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4, eps=layer_norm_eps),
            nn.GELU()
        )

        # Cross-attention from decoder to encoder memory
        self.decoder_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Two-layer GRU for maintaining state across autoregressive steps
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim + hidden_dim // 4,
            hidden_size=hidden_dim,
            num_layers=2,  # Enhanced: Using 2 layers
            batch_first=True
        )

        # Output projections for quantiles
        self.quantile_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2, eps=layer_norm_eps),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(n_quantiles)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for better initial training"""
        for name, p in self.named_parameters():
            if 'weight' in name and len(p.shape) >= 2:
                nn.init.xavier_uniform_(p, gain=0.01)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif 'embedding' in name:
                nn.init.normal_(p, mean=0.0, std=0.01)
            # Initialize pattern bank with small values
            elif 'pattern_bank' in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        values,               # [B, seq_len] scaled inputs
        disease_type,         # [B] long tensor
        target_type,
        population,           # [B] float tensor
        day_indices,          # [B, seq_len] absolute day indices
        valid_mask=None,
        target_values=None
    ):
        """
        Forward pass through the model with enhanced temporal features and pattern memory.

        Args:
            values: Scaled input values [batch_size, seq_len]
            disease_type: Disease type indices [batch_size]
            time_series_type: Time series type indices [batch_size] (0=cases, 1=hosp, 2=death)
            population: Scaled population values [batch_size]
            day_indices: Day indices for temporal patterns [batch_size, seq_len]
            valid_mask: Mask for padding [batch_size, seq_len], True for valid
            target_values: Optional targets for teacher forcing [batch_size, forecast_horizon]

        Returns:
            predictions: Quantile predictions [batch_size, forecast_horizon, n_quantiles]
        """
        batch_size, seq_len, _ = values.shape
        device = values.device

        # ===== INPUT PROCESSING =====
        # Process value data with multi-scale CNN
        value_features = self.values_embedding(values)

        # Process disease type
        disease_emb = self.disease_embedding(disease_type)  # [B, disease_dim]
        disease_emb = self.disease_norm(disease_emb)


        # Process population
        pop_emb = self.population_mlp(population.unsqueeze(-1))  # [B, pop_dim]

        # Enhanced temporal embeddings from day indices
        # 1. Day of week (0-6)
        day_of_week = (day_indices % 7).long()  # [B, seq_len]
        dow_emb = self.day_of_week_embed(day_of_week)  # [B, seq_len, 128]
        dow_emb = self.day_norm(dow_emb)

        # 2. Month approximation (0-11) by dividing day index by 30
        month_approx = ((day_indices // 30) % 12).long()  # [B, seq_len]
        month_emb = self.month_embed(month_approx)  # [B, seq_len, 128]
        month_emb = self.month_norm(month_emb)

        # 3. Day of year approximation (0-365)
        day_of_year = (day_indices % 366).long()  # [B, seq_len]
        doy_emb = self.day_of_year_embed(day_of_year)  # [B, seq_len, 128]
        doy_emb = self.day_of_year_norm(doy_emb)

        # Expand static features to time dimension
        disease_emb_exp = disease_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        pop_emb_exp = pop_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

        target_type_emb = self.target_type_embedding(target_type)
        target_type_emb = self.target_type_norm(target_type_emb)
        target_type_emb_exp = target_type_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

        # Concatenate all features
        combined_features = torch.cat([
            value_features, disease_emb_exp, pop_emb_exp,
            target_type_emb_exp, dow_emb, month_emb, doy_emb
        ], dim=-1)

        # Project to hidden dimension
        encoder_input = self.feature_projection(combined_features)
        encoder_input = self.feature_norm(encoder_input)

        # ===== ENCODER =====
        encoder_output = encoder_input
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output, valid_mask)

        # ===== PATTERN MEMORY =====
        # Apply pattern memory after encoder blocks
        encoder_output = self.pattern_memory(encoder_output, valid_mask)

        # ===== AUTOREGRESSIVE DECODER =====
        # Get initial decoder state (weighted pooling of encoder outputs)
        if valid_mask is not None:
            # Perform masked pooling
            mask_expanded = valid_mask.unsqueeze(-1).float()
            weighted_sum = (encoder_output * mask_expanded).sum(dim=1)
            mask_sum = mask_expanded.sum(dim=1) + 1e-10
            pooled_state = weighted_sum / mask_sum
        else:
            # Simple mean pooling
            pooled_state = encoder_output.mean(dim=1)  # [B, hidden_dim]

        # Initialize decoder state for 2-layer GRU
        dec_hidden = self.decoder_init_proj(pooled_state)
        # Replicate the same initial state for both GRU layers
        dec_hidden = dec_hidden.unsqueeze(0).repeat(2, 1, 1)  # [2, B, hidden_dim] for 2-layer GRU

        # Initial input (last value from the sequence)
        if seq_len > 0:
            decoder_input = values[:, -1, 0].unsqueeze(-1)
        else:
            decoder_input = torch.zeros(batch_size, 1, device=device)

        # Storage for predictions
        all_quantile_preds = []

        # Determine teacher forcing (only during training)
        use_teacher_forcing = (
            self.training and
            target_values is not None and
            torch.rand(1).item() < self.teacher_forcing_ratio
        )

        # Generate predictions autoregressively
        for t in range(self.forecast_horizon):
            # Embed current input - this should produce [B, hidden_dim//4]
            dec_input_emb = self.decoder_input_proj(decoder_input.unsqueeze(-1))  # [B, hidden_dim//4]

            # Attend to encoder outputs - shape: [B, 1, hidden_dim]
            # Use the top layer of GRU output for attention query
            query = dec_hidden[-1:].transpose(0, 1)  # [B, 1, hidden_dim]
            attn_output, _ = self.decoder_attn(
                query, encoder_output, encoder_output,
                key_padding_mask=None if valid_mask is None else ~valid_mask
            )

            # Ensure dimensions match for concatenation by adding a sequence dimension to dec_input_emb if needed
            # Reshape dec_input_emb to [B, 1, hidden_dim//4]
            if len(dec_input_emb.shape) == 2:
                dec_input_emb = dec_input_emb.unsqueeze(1)

            # Now concatenate along feature dimension
            gru_input = torch.cat([attn_output, dec_input_emb], dim=-1)  # [B, 1, hidden_dim+hidden_dim//4]

            # Update decoder state through 2-layer GRU
            _, dec_hidden = self.decoder_gru(gru_input, dec_hidden)

            # Generate quantile predictions using the top GRU layer's output
            dec_output = dec_hidden[-1, :, :]  # [B, hidden_dim]
            quantile_preds = []
            for q_proj in self.quantile_projections:
                q_pred = q_proj(dec_output)  # [B, 1]
                quantile_preds.append(q_pred)

            # Stack quantiles
            step_pred = torch.cat(quantile_preds, dim=1)  # [B, n_quantiles]
            all_quantile_preds.append(step_pred.unsqueeze(1))  # [B, 1, n_quantiles]

            # Determine next input
            median_idx = self.n_quantiles // 2
            if use_teacher_forcing and t < target_values.size(1):
                decoder_input = target_values[:, t].unsqueeze(-1)
            else:
                decoder_input = step_pred[:, median_idx].unsqueeze(-1)

        # Stack predictions across time steps
        predictions = torch.cat(all_quantile_preds, dim=1)  # [B, forecast_horizon, n_quantiles]
        return predictions

    def predict(
        self,
        values,
        disease_type,
        target_type,
        population,
        day_indices,
        valid_mask=None
    ):
        """Inference helper with no teacher forcing"""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            preds = self.forward(
                values,
                disease_type,
                target_type,
                population,
                day_indices,
                valid_mask=valid_mask,
                target_values=None
            )
        if was_training:
            self.train()
        return preds

################################################################################
## PART 3: TRAINING AND EVALUATION FUNCTIONS
################################################################################

def interval_score(lower, upper, observed, alpha):
    """
    Computes the Interval Score (IS) for a single prediction interval.

    Args:
        lower: Lower bound of prediction interval [B, forecast_horizon]
        upper: Upper bound of prediction interval [B, forecast_horizon]
        observed: Observed values [B, forecast_horizon]
        alpha: Decimal value indicating how much is outside the interval (e.g., 0.1 for 90% interval)

    Returns:
        Interval score [B, forecast_horizon]
    """
    # Width of the interval (dispersion component)
    width = upper - lower

    # Underprediction penalty: (lower - observed) * I(observed < lower)
    underprediction = torch.clamp(lower - observed, min=0.0)

    # Overprediction penalty: (observed - upper) * I(observed > upper)
    overprediction = torch.clamp(observed - upper, min=0.0)

    # Interval score = width + (2/alpha) * (underprediction + overprediction)
    is_score = width + (2.0 / alpha) * (underprediction + overprediction)

    return is_score

def weighted_interval_score(preds, target, quantiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99], count_median_twice=False):
    """
    Computes the Weighted Interval Score (WIS) for multiple quantiles.

    Args:
        preds: Tensor of shape [B, forecast_horizon, n_quantiles]
        target: Tensor of shape [B, forecast_horizon]
        quantiles: List of quantile levels
        count_median_twice: Whether to count the median twice in the score

    Returns:
        WIS score (scalar)
    """
    batch_size, forecast_horizon, n_quantiles = preds.shape
    device = preds.device

    # Find median index
    median_idx = n_quantiles // 2  # For 23 quantiles, this is index 11 (0.5 quantile)

    # Get median predictions
    median_preds = preds[..., median_idx]  # [B, forecast_horizon]

    # Compute median component: |y - m|
    median_component = torch.abs(target - median_preds)

    # Find symmetric prediction intervals
    # For 23 quantiles: (0.01,0.99), (0.025,0.975), (0.05,0.95), (0.1,0.9), (0.15,0.85),
    # (0.2,0.8), (0.25,0.75), (0.3,0.7), (0.35,0.65), (0.4,0.6), (0.45,0.55)
    n_intervals = (n_quantiles - 1) // 2  # 11 intervals for 23 quantiles

    interval_scores = []
    weights = []

    for k in range(n_intervals):
        # Get lower and upper quantile indices
        lower_idx = k
        upper_idx = n_quantiles - 1 - k

        # Get alpha for this interval (how much is outside)
        alpha = 1.0 - (quantiles[upper_idx] - quantiles[lower_idx])

        # Compute interval score
        lower_bound = preds[..., lower_idx]
        upper_bound = preds[..., upper_idx]

        is_score = interval_score(lower_bound, upper_bound, target, alpha)
        interval_scores.append(is_score)

        # Weight is alpha/2
        weights.append(alpha / 2.0)

    # Stack interval scores: [n_intervals, B, forecast_horizon]
    interval_scores = torch.stack(interval_scores, dim=0)
    weights = torch.tensor(weights, device=device).view(-1, 1, 1)  # [n_intervals, 1, 1]

    # Weighted sum of interval scores
    weighted_intervals = (weights * interval_scores).sum(dim=0)  # [B, forecast_horizon]

    # Total weight for normalization
    K = n_intervals
    total_weight = K + 0.5

    # Combine median and interval components
    if count_median_twice:
        # Weight median as 1.0 (like a prediction interval)
        median_weight = 1.0
    else:
        # Weight median as 0.5 (like a single quantile)
        median_weight = 0.5

    # Final WIS calculation
    wis = (1.0 / total_weight) * (median_weight * median_component + weighted_intervals)

    return wis.mean()

def quantile_loss(preds, target, quantiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]):
    """
    Computes the average quantile loss over all given quantiles.

    Args:
        preds: Tensor of shape [B, forecast_horizon, n_quantiles]
        target: Tensor of shape [B, forecast_horizon]
        quantiles: list of floats (e.g. [0.05, 0.5, 0.95])
    Returns:
        scalar loss (mean over batch, horizon, quantiles)
    """
    # Expand target from [B, fh] to [B, fh, 1] for broadcasting
    # We'll define e = target - preds
    errors = target.unsqueeze(-1) - preds  # shape: [B, fh, n_quantiles]

    all_losses = []
    for i, q in enumerate(quantiles):
        e = errors[..., i]  # [B, fh] for this quantile
        # Pinball loss for quantile q
        # L_q(e) = max(qe, (q-1)e)
        # (since e = y - y_pred, if e>0 => cost = qe, else cost = (q-1)e)
        loss_q = torch.max(q * e, (q - 1) * e)
        all_losses.append(loss_q)

    # Stack and mean over quantiles dimension
    # each loss_q is shape [B, fh], so stack => [n_quantiles, B, fh]
    all_losses = torch.stack(all_losses, dim=0)
    # Average across all dimensions: n_quantiles, batch, horizon
    return all_losses.mean()

def apply_missing_data_augmentation(values, missing_prob=0.05):
    """
    Apply missing data augmentation by randomly setting some values to zero.

    Args:
        values: Input tensor of shape [B, seq_len, features]
        missing_prob: Probability of setting a value to zero (default 0.05 = 5%)

    Returns:
        Augmented tensor with some values set to zero
    """
    if not values.requires_grad:
        # Only apply augmentation during training
        return values

    # Create a mask for values to be set to zero
    # Shape: [B, seq_len, features]
    mask = torch.rand_like(values) > missing_prob

    # Apply mask: set selected values to zero
    augmented_values = values * mask.float()

    return augmented_values

def combined_mae_wis_loss(preds, target, quantiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99], mae_weight=0.5, wis_weight=0.5):
    """
    Combined loss function using both MAE and WIS.

    Args:
        preds: Tensor of shape [B, forecast_horizon, n_quantiles]
        target: Tensor of shape [B, forecast_horizon]
        quantiles: List of quantile levels
        mae_weight: Weight for MAE component (default 0.5)
        wis_weight: Weight for WIS component (default 0.5)

    Returns:
        Combined loss (scalar)
    """
    # Get median predictions for MAE calculation
    median_idx = len(quantiles) // 2  # For 23 quantiles, this is index 11
    median_preds = preds[..., median_idx]  # [B, forecast_horizon]

    # Calculate MAE on median predictions
    mae_loss = torch.mean(torch.abs(target - median_preds))

    # Calculate WIS loss
    wis_loss = weighted_interval_score(preds, target, quantiles)

    # Combine losses with weights
    combined_loss = mae_weight * mae_loss + wis_weight * wis_loss

    return combined_loss


def randomly_pick_time_series_by_type(data_dir: str, ts_type: int = 0) -> Tuple[np.ndarray, int]:
    """
    Randomly pick one CSV file from data_dir, then randomly pick one row
    in that CSV, and return its full time series (cases, hosp, or deaths)
    based on ts_type as a 1D numpy array.

    Args:
        data_dir: Directory containing the CSV files
        ts_type: Type of time series (0=cases, 1=hosp, 2=deaths)

    Returns:
        Tuple of (time_series, row_index)
    """
    # Map ts_type to column prefix
    prefix_map = {0: 'cases_day_', 1: 'hosp_day_', 2: 'death_day_'}
    prefix = prefix_map.get(ts_type, 'cases_day_')

    # 1) Gather all CSV files matching "disease_dataset_*.csv"
    csv_files = glob.glob(os.path.join(data_dir, "disease_dataset_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} with pattern 'disease_dataset_*.csv'.")

    # 2) Randomly pick one CSV file
    random_csv = random.choice(csv_files)

    # 3) Read the CSV into a DataFrame
    df = pd.read_csv(random_csv)
    if df.empty:
        raise ValueError(f"The selected CSV ({random_csv}) is empty.")

    # 4) Identify relevant columns (e.g. "cases_day_0", "hosp_day_1", "death_day_2" etc.)
    def extract_day_index(col):
        # For "cases_day_12", this returns 12
        return int(col.split('_')[-1])

    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns starting with '{prefix}' found in {random_csv}.")

    # Sort columns by their day index
    cols = sorted(cols, key=extract_day_index)

    # 5) Randomly pick one row index
    row_idx = random.randrange(len(df))

    # 6) Extract that row's values and convert to a NumPy array
    row_series = df.iloc[row_idx][cols].values.astype('float32')

    ts_type_names = {0: "cases", 1: "hospitalizations", 2: "deaths"}
    ts_name = ts_type_names.get(ts_type, "unknown")

    print(f"Selected file: {random_csv}")
    print(f"Selected row index: {row_idx} (out of {len(df)})")
    print(f"Time series type: {ts_name}")

    return row_series, row_idx

def calculate_and_print_metrics(forecasts, baselines, actuals, ts_type=0):
    """
    Helper function to calculate and print metrics for evaluations

    Args:
        forecasts: List of forecast arrays
        baselines: List of baseline forecast arrays
        actuals: List of actual value arrays
        ts_type: Type of time series (0=cases, 1=hosp, 2=deaths) for labeling
    """
    ts_type_names = {0: "Cases", 1: "Hospitalizations", 2: "Deaths"}
    ts_name = ts_type_names.get(ts_type, "Values")

    model_mae_list = []
    model_mse_list = []
    baseline_mae_list = []
    baseline_mse_list = []

    # Loop over each forecast window
    for pred_raw, baseline_raw, actual_raw in zip(forecasts, baselines, actuals):
        model_mae_list.append(np.mean(np.abs(pred_raw - actual_raw)))
        model_mse_list.append(np.mean((pred_raw - actual_raw) ** 2))
        baseline_mae_list.append(np.mean(np.abs(baseline_raw - actual_raw)))
        baseline_mse_list.append(np.mean((baseline_raw - actual_raw) ** 2))

    avg_model_mae = np.mean(model_mae_list)
    avg_baseline_mae = np.mean(baseline_mae_list)
    avg_model_mse = np.mean(model_mse_list)
    avg_baseline_mse = np.mean(baseline_mse_list)

    if avg_baseline_mae != 0:
        rel_improvement_mae = (avg_baseline_mae - avg_model_mae) / avg_baseline_mae * 100
    else:
        rel_improvement_mae = 0.0

    if avg_baseline_mse != 0:
        rel_improvement_mse = (avg_baseline_mse - avg_model_mse) / avg_baseline_mse * 100
    else:
        rel_improvement_mse = 0.0

    print(f"  {ts_name} Model MAE: {avg_model_mae:.4f}, Baseline MAE: {avg_baseline_mae:.4f}, "
          f"Improvement: {rel_improvement_mae:.2f}%")
    print(f"  {ts_name} Model MSE: {avg_model_mse:.4f}, Baseline MSE: {avg_baseline_mse:.4f}, "
          f"Improvement: {rel_improvement_mse:.2f}%")

def evaluate_on_us_states(
    model,
    data_dir,
    device,
    global_stats,
    target_type=2,           # 0=cases, 1=hosp, 2=death (default: death)
    covariate_type=1,        # 0=cases, 1=hosp, 2=death (default: hosp)
    forecast_horizon: int = 4,
    min_context: int = 8,
    plot_random_state: bool = True
):
    """
    Evaluate forecasting model for any target/covariate combination on US states data.

    Args:
        model: Trained forecasting model
        data_dir: Directory containing CSV files (cases.csv, hospitalizations.csv, deaths.csv)
        device: Device to run evaluation on
        global_stats: Dictionary with normalization stats for each time series type
        target_type: What to predict (0=cases, 1=hosp, 2=death)
        covariate_type: What to use as covariate (must be different from target_type)
        forecast_horizon: Number of weeks to forecast
        min_context: Minimum context length required
        plot_random_state: Whether to plot results for a random state

    Returns:
        Dictionary with evaluation results
    """
    if target_type == covariate_type:
        raise ValueError("target_type and covariate_type must be different")

    model.eval()

    # --------------------------------------------------
    # 1. Load the appropriate CSV files
    # --------------------------------------------------
    type_to_file = {0: 'cases.csv', 1: 'hospitalizations.csv', 2: 'deaths.csv'}
    type_to_name = {0: 'cases', 1: 'hospitalizations', 2: 'deaths'}
    type_to_stats = {
        0: ('cases_mean', 'cases_std'),
        1: ('hosp_mean', 'hosp_std'),
        2: ('death_mean', 'death_std')
    }

    target_file = os.path.join(data_dir, type_to_file[target_type])
    cov_file = os.path.join(data_dir, type_to_file[covariate_type])

    if not (os.path.exists(target_file) and os.path.exists(cov_file)):
        missing_files = []
        if not os.path.exists(target_file):
            missing_files.append(type_to_file[target_type])
        if not os.path.exists(cov_file):
            missing_files.append(type_to_file[covariate_type])
        raise FileNotFoundError(f"Missing files in {data_dir}: {missing_files}")

    df_target = pd.read_csv(target_file)
    df_cov = pd.read_csv(cov_file)

    if df_target.shape != df_cov.shape:
        raise ValueError(f"Target ({type_to_file[target_type]}) and covariate ({type_to_file[covariate_type]}) CSVs differ in shape – they must be aligned")

    # Valid state columns (lower-case for robustness)
    state_cols = [c for c in df_target.columns
                  if c.lower() in {'ak','al','ar','az','ca','co','ct','dc','de','fl','ga',
                                   'hi','ia','id','il','in','ks','ky','la','ma','md','me',
                                   'mi','mn','mo','ms','mt','nc','nd','ne','nh','nj','nm',
                                   'nv','ny','oh','ok','or','pa','pr','ri','sc','sd','tn',
                                   'tx','ut','va','vt','wa','wi','wv','wy'}]
    if not state_cols:
        raise ValueError(f"No recognised state columns in {type_to_file[target_type]}")

    print(f"Found {len(state_cols)} state columns")
    print(f"Evaluating: predicting {type_to_name[target_type]} from {type_to_name[target_type]} + {type_to_name[covariate_type]}")

    # --------------------------------------------------
    # 2. Get normalization stats
    # --------------------------------------------------
    target_mean_key, target_std_key = type_to_stats[target_type]
    cov_mean_key, cov_std_key = type_to_stats[covariate_type]

    target_mean, target_std = global_stats[target_mean_key], global_stats[target_std_key]
    cov_mean, cov_std = global_stats[cov_mean_key], global_stats[cov_std_key]

    # --------------------------------------------------
    # 3. Evaluation setup
    # --------------------------------------------------
    random_state = random.choice(state_cols) if plot_random_state else None
    plot_buf = dict(
        historical_target=None,
        historical_cov=None,
        forecasts=[],
        actuals=[],
        q05=[],
        q95=[],
        starts=[],
        state=random_state,
        target_name=type_to_name[target_type],
        cov_name=type_to_name[covariate_type]
    )

    mae_per_state, cov_per_state, wis_per_state, ql_per_state = {}, {}, {}, {}
    all_mae, all_cov, all_wis, all_ql = [], [], [], []

    # --------------------------------------------------
    # 4. Per-state evaluation
    # --------------------------------------------------
    for state in state_cols:
        target_data = df_target[state].values.astype('float32')
        cov_data = df_cov[state].values.astype('float32')

        if len(target_data) < min_context + forecast_horizon:
            print(f"{state}: insufficient length, skipping")
            continue

        if plot_random_state and state == random_state:
            plot_buf['historical_target'] = target_data
            plot_buf['historical_cov'] = cov_data

        seq_mae, seq_cov, seq_wis, seq_ql = [], [], [], []

        # Rolling evaluation
        max_start = len(target_data) - forecast_horizon
        for start in range(min_context, max_start):
            # Context windows
            target_ctx = target_data[:start]
            cov_ctx = cov_data[:start]
            target_future = target_data[start:start + forecast_horizon]

            # Scale each channel with appropriate stats
            target_scaled = (np.log1p(target_ctx) - target_mean) / target_std
            cov_scaled = (np.log1p(cov_ctx) - cov_mean) / cov_std

            # Stack as [seq_len, 2] - [target, covariate]
            x = np.stack([target_scaled, cov_scaled], 1)

            # Convert to tensors
            inp = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.ones(inp.size(0), inp.size(1), dtype=torch.bool, device=device)
            weeks = (torch.arange(start - len(target_ctx), start, device=device) + 1098).unsqueeze(0)
            disease_type = torch.tensor([0], dtype=torch.long, device=device)
            target_type_tensor = torch.tensor([target_type], dtype=torch.long, device=device)
            population = torch.tensor([0.0], dtype=torch.float32, device=device)  # dummy

            with torch.no_grad():
                q_preds = model.predict(
                    inp,
                    disease_type,
                    target_type_tensor,  # New: pass target type
                    population,
                    weeks,
                    mask
                )  # [1, H, Q]
                q_preds = q_preds[:, :forecast_horizon, :].squeeze(0).cpu().numpy()

            # Extract quantiles (assuming 23 quantiles: 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)
            med_scaled, q05_scaled, q95_scaled = q_preds[:, 11], q_preds[:, 2], q_preds[:, 20]

            # Denormalize predictions using target stats
            med_raw = np.expm1(med_scaled * target_std + target_mean)
            q05_raw = np.expm1(q05_scaled * target_std + target_mean)
            q95_raw = np.expm1(q95_scaled * target_std + target_mean)

            # Calculate metrics
            seq_mae.append(np.mean(np.abs(med_raw - target_future)))
            seq_cov.append(np.mean((target_future >= q05_raw) & (target_future <= q95_raw)))

            # Calculate WIS for this prediction using raw data (like MAE)
            # Denormalize all quantile predictions to raw space
            preds_raw = np.expm1(q_preds * target_std + target_mean)  # [forecast_horizon, n_quantiles]
            preds_raw_tensor = torch.tensor(preds_raw, dtype=torch.float32, device=device)
            target_raw_tensor = torch.tensor(target_future, dtype=torch.float32, device=device)

            # Calculate WIS using raw data
            wis_score = weighted_interval_score(
                preds_raw_tensor.unsqueeze(0),
                target_raw_tensor.unsqueeze(0),
                quantiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
            )
            seq_wis.append(wis_score.item())

            # Calculate Quantile Loss using raw data
            ql_score = quantile_loss(
                preds_raw_tensor.unsqueeze(0),
                target_raw_tensor.unsqueeze(0),
                quantiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
            )
            seq_ql.append(ql_score.item())

            # Store for plotting
            if plot_random_state and state == random_state:
                plot_buf['forecasts'].append(med_raw)
                plot_buf['actuals'].append(target_future)
                plot_buf['q05'].append(q05_raw)
                plot_buf['q95'].append(q95_raw)
                plot_buf['starts'].append(start)

        if seq_mae:
            mae = float(np.mean(seq_mae))
            cover = float(np.mean(seq_cov))
            wis = float(np.mean(seq_wis))
            ql = float(np.mean(seq_ql))
            mae_per_state[state] = mae
            cov_per_state[state] = cover
            wis_per_state[state] = wis
            ql_per_state[state] = ql
            all_mae.append(mae)
            all_cov.append(cover)
            all_wis.append(wis)
            all_ql.append(ql)
            print(f"{state}: MAE={mae:.4f}, QL={ql:.4f}, WIS={wis:.4f}, 90%-cov={cover*100:.1f}%")

    overall_mae = np.mean(all_mae) if all_mae else float('nan')
    overall_cov = np.mean(all_cov) if all_cov else float('nan')
    overall_wis = np.mean(all_wis) if all_wis else float('nan')
    overall_ql = np.mean(all_ql) if all_ql else float('nan')

    print(f"\nUS-state {type_to_name[target_type].upper()} forecast from {type_to_name[covariate_type]} – overall MAE: {overall_mae:.4f}, QL: {overall_ql:.4f}, WIS: {overall_wis:.4f}, 90% coverage: {overall_cov*100:.1f}%")

    # --------------------------------------------------
    # 5. Optional plotting
    # --------------------------------------------------
    if plot_random_state and plot_buf['historical_target'] is not None:
        plt.figure(figsize=(16, 8))
        weeks = np.arange(len(plot_buf['historical_target']))

        plt.plot(weeks, plot_buf['historical_target'], 'k-', alpha=0.25,
                label=f'{plot_buf["target_name"].title()} (historical)')
        plt.plot(weeks, plot_buf['historical_cov'], 'c-', alpha=0.25,
                label=f'{plot_buf["cov_name"].title()} (historical)')

        for i, (fc, act, q05, q95, start) in enumerate(zip(
            plot_buf['forecasts'], plot_buf['actuals'],
            plot_buf['q05'], plot_buf['q95'],
            plot_buf['starts'])):

            rng = range(start, start + len(fc))
            plt.plot(rng, fc, 'r-', alpha=0.6, label='Forecast median' if i == 0 else None)
            plt.fill_between(rng, q05, q95, color='red', alpha=0.15,
                           label='90% CI' if i == 0 else None)
            plt.plot(rng, act, 'b-', alpha=0.8, label='Actual' if i == 0 else None)
            plt.axvline(start, color='gray', linestyle='--', alpha=0.2)

        plt.title(f"All rolling forecasts – {plot_buf['state']} ({plot_buf['target_name']} from {plot_buf['cov_name']})")
        plt.xlabel("Week index")
        plt.ylabel(f"Weekly {plot_buf['target_name']}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    model.train()
    return dict(
        mae_per_state=mae_per_state,
        coverage_per_state=cov_per_state,
        wis_per_state=wis_per_state,
        ql_per_state=ql_per_state,
        overall_mae=overall_mae,
        overall_coverage=overall_cov,
        overall_wis=overall_wis,
        overall_ql=overall_ql,
        target_type=target_type,
        covariate_type=covariate_type
    )



def evaluate_on_mi_regions(
    model,
    data_dir,
    device,
    global_stats,
    forecast_horizon=8,
    min_context=112,
    plot_results=True,
    save_plots=False,
    plot_dir=None,
    plot_random_region=True
):
    """
    Evaluate the model on Michigan regions flu hospitalization data.
    Calculates both MAE and 90% interval coverage rate.
    If plot_random_region=True, randomly selects one region and plots all forecasts.
    """
    model.eval()

    csv_path = os.path.join(data_dir, "flu_hosp.csv")
    if not os.path.exists(csv_path):
        print(f"flu_hosp.csv not found at {csv_path}, skipping Michigan regions evaluation")
        return {}

    df = pd.read_csv(csv_path)

    region_populations = {
        'Region 1': 1083652, 'Region 2N': 2306347, 'Region 2S': 2253888, 'Region 3': 1074908,
        'Region 5': 972828, 'Region 6': 1556207, 'Region 7': 349801, 'Region 8': 279802
    }

    region_cols = [col for col in df.columns if col in region_populations]

    if not region_cols:
        print("No region columns found in flu_hosp.csv")
        return {}

    print(f"Found {len(region_cols)} Michigan region columns")

    global_mean = global_stats['hosp_mean']
    global_std = global_stats['hosp_std']
    pop_mean, pop_std = 14.1607, 1.9670  # Same as training

    # Randomly select one region for plotting
    random_region = random.choice(region_cols) if plot_random_region else None
    plot_data = {
        'historical_data': None,
        'all_forecasts': [],
        'all_actuals': [],
        'all_q05': [],
        'all_q95': [],
        'forecast_starts': [],
        'region_name': random_region
    }

    mae_per_region = {}
    coverage_per_region = {}
    wis_per_region = {}
    ql_per_region = {}
    all_maes = []
    all_coverages = []
    all_wis = []
    all_ql = []

    for region in region_cols:
        weekly_data = df[region].values.astype('float32')

        if len(weekly_data) < min_context + forecast_horizon:
            print(f"Region {region} has insufficient data (length {len(weekly_data)})")
            continue

        # Store data for plotting if this is the selected region
        if plot_random_region and region == random_region:
            plot_data['historical_data'] = weekly_data

        region_population = region_populations[region]
        forecasts = []
        actuals = []
        coverages = []
        wis_scores = []
        ql_scores = []
        q05_forecasts = []
        q95_forecasts = []
        forecast_starts = []

        max_idx = len(weekly_data) - forecast_horizon

        for start_idx in range(min_context, max_idx):
            historical_data = weekly_data[:start_idx]
            future_data = weekly_data[start_idx:start_idx+forecast_horizon]

            log_historical = np.log1p(historical_data)
            scaled_historical = (log_historical - global_mean) / global_std

            context_length = len(historical_data)
            week_indices_arr = np.arange(start_idx - context_length, start_idx, dtype=np.int64) + 1098
            week_indices_tensor = torch.tensor(week_indices_arr, dtype=torch.long, device=device).unsqueeze(0)

            input_tensor = torch.tensor(scaled_historical, dtype=torch.float32, device=device).unsqueeze(0)
            valid_mask = torch.ones_like(input_tensor, dtype=torch.bool)

            disease_type = torch.tensor([0], dtype=torch.long, device=device)
            ts_type_tensor = torch.tensor([1], dtype=torch.long, device=device)

            pop_log = np.log1p(region_population)
            scaled_pop = (pop_log - pop_mean) / pop_std
            population = torch.tensor([scaled_pop], dtype=torch.float32, device=device)

            with torch.no_grad():
                pred_quantiles = model.predict(
                    input_tensor,
                    disease_type,
                    ts_type_tensor,
                    population,
                    week_indices_tensor,
                    valid_mask=valid_mask
                )

            pred_all_quantiles = pred_quantiles.squeeze(0).cpu().numpy()
            pred_median_scaled = pred_all_quantiles[:, 11]  # 0.5 quantile (index 11 of 23)
            pred_q05_scaled = pred_all_quantiles[:, 2]     # 0.05 quantile (index 2 of 23)
            pred_q95_scaled = pred_all_quantiles[:, 20]     # 0.95 quantile (index 20 of 23)

            pred_median_raw = np.expm1(pred_median_scaled * global_std + global_mean)
            pred_q05_raw = np.expm1(pred_q05_scaled * global_std + global_mean)
            pred_q95_raw = np.expm1(pred_q95_scaled * global_std + global_mean)

            forecasts.append(pred_median_raw)
            actuals.append(future_data)
            q05_forecasts.append(pred_q05_raw)
            q95_forecasts.append(pred_q95_raw)
            forecast_starts.append(start_idx)

            # Store data for plotting if this is the selected region
            if plot_random_region and region == random_region:
                plot_data['all_forecasts'].append(pred_median_raw)
                plot_data['all_actuals'].append(future_data)
                plot_data['all_q05'].append(pred_q05_raw)
                plot_data['all_q95'].append(pred_q95_raw)
                plot_data['forecast_starts'].append(start_idx)

            coverage_mask = (future_data >= pred_q05_raw) & (future_data <= pred_q95_raw)
            coverages.append(coverage_mask.astype(float))

            # Calculate WIS for this prediction using raw data (like MAE)
            # Denormalize all quantile predictions to raw space
            preds_raw = np.expm1(pred_all_quantiles * global_std + global_mean)  # [forecast_horizon, n_quantiles]
            preds_raw_tensor = torch.tensor(preds_raw, dtype=torch.float32, device=device)
            target_raw_tensor = torch.tensor(future_data, dtype=torch.float32, device=device)

            wis_score = weighted_interval_score(
                preds_raw_tensor.unsqueeze(0),
                target_raw_tensor.unsqueeze(0),
                quantiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
            )
            wis_scores.append(wis_score.item())

            # Calculate Quantile Loss using raw data
            ql_score = quantile_loss(
                preds_raw_tensor.unsqueeze(0),
                target_raw_tensor.unsqueeze(0),
                quantiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
            )
            ql_scores.append(ql_score.item())

        forecasts = np.concatenate(forecasts)
        actuals = np.concatenate(actuals)
        coverages = np.concatenate(coverages)

        mae = np.mean(np.abs(forecasts - actuals))
        coverage = np.mean(coverages)
        wis = np.mean(wis_scores)
        ql = np.mean(ql_scores)

        mae_per_region[region] = mae
        coverage_per_region[region] = coverage
        wis_per_region[region] = wis
        ql_per_region[region] = ql
        all_maes.append(mae)
        all_coverages.append(coverage)
        all_wis.append(wis)
        all_ql.append(ql)

        print(f"{region}: MAE = {mae:.4f}, QL = {ql:.4f}, WIS = {wis:.4f}, 90% Coverage = {coverage * 100:.2f}%")

    overall_mae = np.mean(all_maes) if all_maes else float('nan')
    overall_coverage = np.mean(all_coverages) if all_coverages else float('nan')
    overall_wis = np.mean(all_wis) if all_wis else float('nan')
    overall_ql = np.mean(all_ql) if all_ql else float('nan')

    print(f"\nMichigan Regions Overall MAE: {overall_mae:.4f}")
    print(f"Michigan Regions Overall QL: {overall_ql:.4f}")
    print(f"Michigan Regions Overall WIS: {overall_wis:.4f}")
    print(f"Michigan Regions Overall 90% Coverage: {overall_coverage * 100:.2f}%")

    # Create plot for randomly selected region
    if plot_random_region and plot_data['historical_data'] is not None and plot_data['all_forecasts']:
        plt.figure(figsize=(16, 8))

        # Plot historical data
        historical = plot_data['historical_data']
        plt.plot(range(len(historical)), historical, 'k-', alpha=0.3, linewidth=1.0, label='Historical Data')

        # Plot all forecasts
        for i, (forecast, actual, q05, q95, start_idx) in enumerate(zip(
            plot_data['all_forecasts'],
            plot_data['all_actuals'],
            plot_data['all_q05'],
            plot_data['all_q95'],
            plot_data['forecast_starts']
        )):
            x_range = range(start_idx, start_idx + len(forecast))

            # Plot forecast (median)
            plt.plot(x_range, forecast, 'r-', alpha=0.6, linewidth=1.0,
                    label='Model Forecast' if i == 0 else None)

            # Plot confidence interval
            plt.fill_between(x_range, q05, q95, color='red', alpha=0.15,
                           label='90% Confidence Interval' if i == 0 else None)

            # Plot actual values
            plt.plot(x_range, actual, 'b-', alpha=0.8, linewidth=1.5,
                    label='Actual Values' if i == 0 else None)

            # Add vertical line at forecast start
            plt.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.2)

        plt.title(f'All Forecasts for {plot_data["region_name"]} - Flu Hospitalizations')
        plt.xlabel('Week')
        plt.ylabel('Weekly Flu Hospitalizations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    model.train()

    return {
        'mae_per_region': mae_per_region,
        'coverage_per_region': coverage_per_region,
        'wis_per_region': wis_per_region,
        'ql_per_region': ql_per_region,
        'overall_mae': overall_mae,
        'overall_coverage': overall_coverage,
        'overall_wis': overall_wis,
        'overall_ql': overall_ql
    }

def growth_rate_loss(preds, targets, quantile_idx=4, global_mean=0.0, global_std=1.0):
    """
    Computes MSE of growth rate in raw case space using denormalized log1p values.

    Args:
        preds: [B, H, Q] - quantile predictions in normalized log1p-space
        targets: [B, H] - target values in normalized log1p-space
        quantile_idx: index of the quantile to use (e.g., 4 for median)
        global_mean, global_std: used to denormalize log1p-space to raw case counts
    """
    # Extract median prediction [B, H]
    pred = preds[..., quantile_idx]

    # Denormalize
    pred_log = pred * global_std + global_mean
    target_log = targets * global_std + global_mean

    # Invert log1p to get raw counts
    pred_raw = torch.expm1(pred_log).clamp(min=1e-3)
    target_raw = torch.expm1(target_log).clamp(min=1e-3)

    # Compute log growth rates
    pred_rate = torch.log(pred_raw[:, 1:] / pred_raw[:, :-1])
    target_rate = torch.log(target_raw[:, 1:] / target_raw[:, :-1])

    return F.mse_loss(pred_rate, target_rate)

def train_multi_time_series_forecaster(
    model,
    train_loader,
    optimizer,
    scheduler,
    device='cuda',
    forecast_horizon=4,  # 4-week forecast
    learning_rate=1e-4,
    quantiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
    eval_interval=1000,
    eval_data_dir=None,
    global_stats=None,  # Dictionary with mean/std for each time series type
    save_dir=None,
    gradient_clip_val=1.0,
    initial_teacher_forcing_ratio=None,  # If None, use model's default
    final_teacher_forcing_ratio=0.0,     # Target teacher forcing ratio (0 = no teacher forcing)
    num_epochs=1,
    resume_checkpoint: str | None = None,
    missing_data_prob=0.05,  # Probability of setting values to zero for robustness training
):
    """
    Trains the multi-time-series forecaster model with multiple time series types.
    Periodically evaluates the model and saves checkpoints.

    Features:
    - Teacher forcing ratio linear decay from initial to final (default = 0)
    - Learning rate scheduling
    - Evaluation on all three time series types
    """

    # ----------  resume defaults (will be overwritten if checkpoint is loaded)
    start_epoch      = 0
    start_batch_idx  = 0
    steps_done       = 0

    # Set initial teacher forcing ratio if not provided
    if initial_teacher_forcing_ratio is None:
        initial_teacher_forcing_ratio = model.teacher_forcing_ratio

    # Store original teacher forcing ratio (will be modified during training)
    original_tf_ratio = model.teacher_forcing_ratio

    # Calculate total number of steps for scheduling
    total_steps = len(train_loader) * num_epochs




    # ----------  optional resume ----------
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ck = torch.load(resume_checkpoint, map_location=device)
        print(f"Resuming from {resume_checkpoint}")
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch      = ck['epoch']
        steps_done       = ck['steps_done']
        start_batch_idx  = ck['batch_idx'] + 1
        model.teacher_forcing_ratio = ck.get('teacher_forcing_ratio',
                                             model.teacher_forcing_ratio)


    # Loss function - using quantile loss for training
    criterion = lambda preds, target: quantile_loss(preds, target, quantiles)

    # Move model to device
    model = model.to(device)
    model.train()

    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Training setup
    batch_count = 0
    all_train_losses = []
    best_loss = float('inf')

    gradient_accumulation_steps = 64

    # Main training loop for multiple epochs
    for epoch in range(num_epochs):
        print(f"\nStarting Epoch {epoch+1}/{num_epochs}")

        # Progress tracking
        with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for batch_idx, batch in enumerate(pbar):

                if epoch == start_epoch and batch_idx < start_batch_idx: continue


                batch_count += 1
                steps_done += 1

                # Calculate current teacher forcing ratio (linear decay)
                current_tf_ratio = initial_teacher_forcing_ratio * (1 - steps_done / total_steps) + final_teacher_forcing_ratio * (steps_done / total_steps)

                # Update model's teacher forcing ratio
                model.teacher_forcing_ratio = current_tf_ratio

                # Move data to device
                values = batch['values'].to(device)  # [B, seq_len, features]
                target = batch['target_values'].to(device)  # [B, forecast_horizon]

                # Apply missing data augmentation for robustness training
                values_augmented = apply_missing_data_augmentation(values, missing_data_prob)

                # Forward pass
                pred_quantiles = model(
                    values=values_augmented,
                    disease_type=batch['disease_type'].to(device),
                    target_type=batch['target_type'].to(device),
                    population=batch['population'].to(device),
                    day_indices=batch['day_indices'].to(device),
                    valid_mask=batch['valid_mask'].to(device),
                    target_values=target if current_tf_ratio > 0 else None
                )


                # Compute loss
                loss = criterion(pred_quantiles, target)
                all_train_losses.append(loss.item())

                # Backward pass
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

                    # Update weights
                    optimizer.step()

                    # Zero gradients after update
                    optimizer.zero_grad()

                    # Update learning rate - only update scheduler after actual optimization steps
                    scheduler.step()

                # Get current learning rate
                current_lr = scheduler.get_last_lr()[0]

                # Update progress bar
                pbar.set_postfix({
                    "loss": loss.item(),
                    "tf_ratio": f"{current_tf_ratio:.3f}",
                    "lr": f"{current_lr:.6f}"
                })

                # Save model periodically
                if (steps_done % 5000) == 0 and steps_done > 0:
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

                    # Calculate average loss over the last eval_interval batches
                    recent_losses = all_train_losses[-eval_interval:] if len(all_train_losses) >= eval_interval else all_train_losses
                    avg_loss = np.mean(recent_losses)

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        save_path = os.path.join(save_dir, f"best_model_{timestamp}_4weekcovariate.pt")
                    else:
                        save_path = os.path.join(save_dir, f"forecaster_checkpoint_{timestamp}_4weekcovariate.pt")

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'steps_done': steps_done,
                        'batch_idx': batch_idx,
                        'teacher_forcing_ratio': model.teacher_forcing_ratio
                    }, save_path)

                    print(f"\nModel saved to {save_path} with avg loss: {avg_loss:.4f}, tf_ratio: {current_tf_ratio:.3f}, lr: {current_lr:.6f}")

                # Print progress and evaluate
                if (steps_done % eval_interval) == 0:
                    # Calculate average loss over the last eval_interval batches
                    recent_losses = all_train_losses[-eval_interval:] if len(all_train_losses) >= eval_interval else all_train_losses
                    avg_loss = np.mean(recent_losses)
                    print(f"\n[Epoch {epoch+1}, Batch {batch_idx}] Avg Loss over last {len(recent_losses)} batches: {avg_loss:.4f}, TF: {current_tf_ratio:.3f}, LR: {current_lr:.6f}")

                    # Evaluation on random time series for each type
                    try:
                        model.eval()

                        # Evaluate on each time series type
                        # for ts_type in range(3):  # 0=cases, 1=hosp, 2=deaths
                        #     try:
                        #         # Get a random weekly time series of this type
                        #         daily_ts, row_idx = randomly_pick_time_series_by_type(
                        #             os.path.dirname(save_dir) if save_dir else ".",
                        #             ts_type=ts_type
                        #         )

                        #         # Convert daily to weekly
                        #         weekly_ts = aggregate_to_weekly(daily_ts)

                        #         # Get appropriate global stats for this type
                        #         if global_stats is not None:
                        #             if ts_type == 0:  # Cases
                        #                 ts_global_mean = global_stats['cases_mean']
                        #                 ts_global_std = global_stats['cases_std']
                        #             elif ts_type == 1:  # Hospitalizations
                        #                 ts_global_mean = global_stats['hosp_mean']
                        #                 ts_global_std = global_stats['hosp_std']
                        #             else:  # Deaths
                        #                 ts_global_mean = global_stats['death_mean']
                        #                 ts_global_std = global_stats['death_std']
                        #         else:
                        #             ts_global_mean, ts_global_std = 0.0, 1.0

                        #         # Perform evaluation using our evaluation function
                        #         print(f"\nEvaluating on random {['cases', 'hospitalizations', 'deaths'][ts_type]} time series:")
                        #         ts_name = {0: "Cases", 1: "Hospitalizations", 2: "Deaths"}[ts_type]

                        #         # Use full evaluation function on this time series
                        #         with torch.no_grad():
                        #             results = evaluate_weekly_forecaster(
                        #                 model=model,
                        #                 test_data=weekly_ts,
                        #                 time_series_type=ts_type,
                        #                 device=device,
                        #                 forecast_horizon=forecast_horizon,
                        #                 min_context=8,
                        #                 quantiles=quantiles,
                        #                 global_stats=global_stats,
                        #                 show_plot=True,
                        #                 series_name=ts_name
                        #             )

                        #             # Calculate metrics from results
                        #             forecasts, baselines, actuals, _, _, _ = results
                        #             calculate_and_print_metrics(forecasts, baselines, actuals, ts_type)

                        #     except Exception as eval_error:
                        #         print(f"Error evaluating time series type {ts_type}: {eval_error}")

                        # Evaluate on US states data for all types if directory is provided
                        if eval_data_dir:
                            print("\nEvaluating model on US states data:")
                            results = evaluate_on_us_states(
                                model=model,
                                data_dir=eval_data_dir,
                                device=device,
                                global_stats=global_stats,
                                forecast_horizon=forecast_horizon,
                                min_context=12
                            )

                            # Print results for each type
                            print(f"\nUS-states overall MAE: {results['overall_mae']:.4f}, QL: {results['overall_ql']:.4f}, WIS: {results['overall_wis']:.4f}, 90% coverage: {results['overall_coverage']*100:.2f}%")



                        model.train()
                    except Exception as e:
                        print(f"Error in evaluation: {e}")
                        traceback.print_exc()
                        model.train()

    # Restore original teacher forcing ratio
    # model.teacher_forcing_ratio = original_tf_ratio

    # End of training
    print("Training complete. Total batches:", batch_count)




    return all_train_losses

################################################################################
## PART 4: MAIN EXECUTION
################################################################################

if __name__ == "__main__":
    # 1) Mount Drive if in Colab
    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        data_dir = "/content/drive/My Drive/disease_datasets/"
    else:
        data_dir = "./disease_datasets/"
    print(f"Using data directory: {data_dir}")

    # 2) Prepare model directory on Drive
    model_dir = os.path.join(data_dir, "saved_models_multi")
    os.makedirs(model_dir, exist_ok=True)

    # 3) Compute global stats and create dataloaders
    global_stats = compute_global_stats_by_time_series_type(data_dir)
    pop_mean, pop_std = get_population_stats(data_dir)
    train_loader, _ = create_multitarget_dataloaders(
        data_dir=data_dir,
        global_stats=global_stats,
        target_type=None,
        covariate_type=None,
        max_context_weeks=112,
        min_weeks_to_start=8,
        forecast_horizon=4,
        batch_size=48,
        num_workers=4
    )
    print(f"Train batches: {len(train_loader)}")

    # 4) Device & model init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MultiTimeSeriesForecaster(
        input_window=112,
        forecast_horizon=4,
        hidden_dim=1024,
        ffn_dim=2048,
        n_layers=16,
        n_heads=16,
        n_quantiles=23,
        disease_embed_dim=64,
        pop_embed_dim=64,
        binary_feat_dim=32,
        teacher_forcing_ratio=0.0,
        dropout=0.2
    ).to(device)

    # ----- build optimiser/scheduler in main so we can restore them -----
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = len(train_loader) // 64   # 1 epoch – keep in sync with call below
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )


    # 5) Load existing checkpoint if any
    ckpt_paths = glob.glob(os.path.join(model_dir, "*_4weekcovariate.pt"))
    resume_ckpt = max(ckpt_paths, key=os.path.getctime) if ckpt_paths else None
    if resume_ckpt:
        print(f"Will resume training from {resume_ckpt}")
    else:
        print("No checkpoint found – starting fresh")

    # 6) Invoke training (resume parameters already baked into train function via save_dir)
    train_losses = train_multi_time_series_forecaster(
        model=model,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        forecast_horizon=4,
        learning_rate=1e-4,
        quantiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
        eval_interval=5000,
        eval_data_dir=data_dir,
        global_stats=global_stats,
        save_dir=model_dir,
        gradient_clip_val=1.0,
        initial_teacher_forcing_ratio=0.0,
        final_teacher_forcing_ratio=0.0,
        resume_checkpoint=resume_ckpt,
        num_epochs=1,
        missing_data_prob=0.001
    )
