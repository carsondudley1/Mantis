#!/usr/bin/env python3
import argparse
import subprocess
import sys
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from statsmodels.tsa.api import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

QUANTILES = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95], dtype=float)
MEDIAN_IDX = int(np.where(QUANTILES == 0.5)[0][0])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Mantis/ETS/SARIMAX/LSTM on Thai weekly CSV."
    )
    parser.add_argument(
        "--input-csv",
        default="thai_hfmd.csv",
        help="Input csv generated from your preprocessing pipeline.",
    )
    parser.add_argument("--target-col", default="Cases")
    parser.add_argument("--covariate-col", default="RH")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--start-week", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--lstm-epochs", type=int, default=10)
    parser.add_argument("--output-summary-csv", default="")
    return parser.parse_args()


def ensure_mantis_and_weights(project_root: Path, horizon: int = 4):
    if horizon != 4:
        raise ValueError("This script auto-download currently supports 4-week covariate model.")

    mantis_repo = project_root / "Mantis"
    if not mantis_repo.exists():
        raise FileNotFoundError(f"Expected local Mantis repo at: {mantis_repo}")

    try:
        from mantis import Mantis  # noqa: F401
    except Exception:
        print("Installing local Mantis package...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(mantis_repo)]
        )
        from mantis import Mantis  # noqa: F401

    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "mantis_4w_cov.pt"

    if not model_path.exists():
        print("Downloading mantis_4w_cov.pt...")
        url = (
            "https://github.com/carsondudley1/Mantis/releases/download/"
            "mantis-v1.0/mantis_4w_cov.pt"
        )
        urlretrieve(url, str(model_path))
        print(f"Downloaded model to {model_path}")

    from mantis import Mantis
    return Mantis


def make_gaussian_quantiles(mean, stderr, quantiles):
    q = np.zeros((len(mean), len(quantiles)), dtype=float)
    for h in range(len(mean)):
        for j, qq in enumerate(quantiles):
            q[h, j] = mean[h] + stats.norm.ppf(qq) * stderr[h]
    return np.maximum(q, 0.0)


def ensure_monotonic_quantiles(q_preds):
    out = q_preds.copy()
    for h in range(out.shape[0]):
        out[h, :] = np.sort(out[h, :])
    return out


def compute_window_metrics(true_future, q_preds, baseline_pred):
    median = q_preds[:, MEDIAN_IDX]
    model_abs = np.abs(true_future - median)
    base_abs = np.abs(true_future - baseline_pred)

    low90, up90 = q_preds[:, 0], q_preds[:, 8]
    cov90_hits = np.sum((true_future >= low90) & (true_future <= up90))

    low50, up50 = q_preds[:, 2], q_preds[:, 6]
    cov50_hits = np.sum((true_future >= low50) & (true_future <= up50))

    return model_abs, base_abs, cov90_hits, cov50_hits


def dm_test(loss1, loss2, h=1):
    d = np.array(loss1) - np.array(loss2)
    t_len = len(d)
    if t_len == 0:
        return np.nan

    mean_d = np.mean(d)

    def autocov(x, lag):
        if lag == 0:
            return np.var(x, ddof=0)
        return np.mean((x[:-lag] - mean_d) * (x[lag:] - mean_d))

    var_d = autocov(d, 0)
    for k in range(1, h):
        var_d += 2 * autocov(d, k)

    if var_d <= 1e-8:
        return np.nan

    dm_stat = mean_d / np.sqrt(var_d / t_len)
    return 2 * (1 - stats.norm.cdf(abs(dm_stat)))


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self, preds, target):
        target_exp = target.unsqueeze(-1).expand_as(preds)
        q = self.quantiles.to(preds.device).view(1, 1, -1)
        errors = target_exp - preds
        return torch.max((q - 1) * errors, q * errors).mean()


class LSTMQuantileForecaster(nn.Module):
    def __init__(self, hidden_dim, horizon, n_quantiles, num_layers=2, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + horizon, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * n_quantiles),
        )

    def forward(self, x_hist, x_future_cov, use_bn=True):
        out, _ = self.lstm(x_hist)
        h = out[:, -1, :]
        if use_bn and h.shape[0] > 1:
            h = self.bn(h)
        z = torch.cat([h, x_future_cov], dim=1)
        y = self.head(z)
        return y.view(-1, self.horizon, self.n_quantiles)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = project_root / input_csv

    if not input_csv.exists():
        available = sorted(p.name for p in project_root.glob("thai_*.csv"))
        raise FileNotFoundError(
            f"Input file not found: {input_csv}\nAvailable thai_*.csv files: {available}"
        )

    Mantis = ensure_mantis_and_weights(project_root, horizon=args.horizon)

    df = pd.read_csv(input_csv)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise ValueError("Expected a 'Date' column in input CSV.")

    for col in [args.target_col, args.covariate_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {input_csv.name}")

    if "Location" not in df.columns:
        raise ValueError("Expected a 'Location' column in input CSV.")

    provinces = sorted(df["Location"].dropna().unique())
    if not provinces:
        raise ValueError("No provinces found in input CSV.")

    forecast_horizon = args.horizon
    stride = args.stride
    start_week = args.start_week
    lookback = args.lookback

    # LSTM settings
    lstm_hidden_dim = 64
    lstm_num_layers = 2
    lstm_dropout = 0.2
    lstm_epochs = args.lstm_epochs
    lstm_batch_size = 32
    lstm_lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mantis_model = Mantis(forecast_horizon=forecast_horizon, use_covariate=True, model_dir=str(project_root / "models"))

    def forecast_mantis(y_hist, x_hist):
        pred = mantis_model.predict(
            time_series=y_hist,
            covariate=x_hist,
            target_type=2,
            covariate_type=1,
        )
        return ensure_monotonic_quantiles(pred)

    def forecast_ets_with_covariate(y_hist, x_hist, x_future_proxy):
        x_design = np.column_stack([np.ones(len(x_hist)), x_hist])
        beta, *_ = np.linalg.lstsq(x_design, y_hist, rcond=None)
        a, b = float(beta[0]), float(beta[1])
        y_det_hist = a + b * x_hist
        resid_hist = np.asarray(y_hist - y_det_hist, dtype=float)

        if len(resid_hist) < 8:
            raise ValueError("Not enough history for ETS")

        resid_series = pd.Series(resid_hist)
        use_seasonal = len(resid_hist) >= 2 * 52 and np.nanstd(resid_hist) > 1e-8
        ets_specs = []
        if use_seasonal:
            ets_specs.append(dict(error="add", trend="add", seasonal="add", seasonal_periods=52))
        ets_specs.extend([
            dict(error="add", trend="add", seasonal=None),
            dict(error="add", trend=None, seasonal=None),
        ])

        last_err = None
        for spec in ets_specs:
            try:
                model = ETSModel(resid_series, **spec)
                fit = model.fit(disp=False)
                pred_obj = fit.get_prediction(
                    start=len(resid_hist),
                    end=len(resid_hist) + forecast_horizon - 1,
                )
                resid_mean = np.asarray(pred_obj.predicted_mean, dtype=float)

                in_sample_resid = np.asarray(fit.resid, dtype=float)
                in_sample_resid = in_sample_resid[np.isfinite(in_sample_resid)]
                resid_std = max(np.nanstd(in_sample_resid, ddof=1), 1e-3)
                horizon_se = resid_std * np.sqrt(np.arange(1, forecast_horizon + 1))

                y_det_future = a + b * x_future_proxy
                total_mean = y_det_future + resid_mean
                q_preds = make_gaussian_quantiles(total_mean, horizon_se + 1e-6, QUANTILES)
                return ensure_monotonic_quantiles(q_preds)
            except Exception as err:
                last_err = err

        raise RuntimeError(f"ETS failed for all specs: {last_err}")

    def forecast_sarimax_with_covariate(y_hist, x_hist, x_future_proxy):
        exog_hist = np.asarray(x_hist, dtype=float).reshape(-1, 1)
        exog_future = np.asarray(x_future_proxy, dtype=float).reshape(-1, 1)
        seasonal_order = (1, 0, 0, 52) if len(y_hist) >= 3 * 52 else (0, 0, 0, 0)
        model = SARIMAX(
            y_hist,
            exog=exog_hist,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            trend="n",
            simple_differencing=True,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, method="lbfgs", maxiter=40)
        fc = fit.get_forecast(steps=forecast_horizon, exog=exog_future)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        if hasattr(fc, "se_mean") and fc.se_mean is not None:
            se = np.asarray(fc.se_mean, dtype=float)
        else:
            pred_var = np.asarray(fc.var_pred_mean, dtype=float)
            se = np.sqrt(np.maximum(pred_var, 1e-8))
        q_preds = make_gaussian_quantiles(mean, se + 1e-6, QUANTILES)
        return ensure_monotonic_quantiles(q_preds)

    def forecast_lstm_quantile(y_hist, x_hist, x_future_proxy):
        n = len(y_hist)
        if n < lookback + forecast_horizon + 1:
            raise ValueError("Not enough history for LSTM training window")

        x_hist_all, x_future_cov_all, y_all = [], [], []
        for i in range(lookback, n - forecast_horizon + 1):
            hist_target = y_hist[i - lookback:i]
            hist_cov = x_hist[i - lookback:i]
            fut_cov = x_hist[i:i + forecast_horizon]
            fut_target = y_hist[i:i + forecast_horizon]
            if len(fut_target) == forecast_horizon:
                x_hist_all.append(np.column_stack([hist_target, hist_cov]))
                x_future_cov_all.append(fut_cov)
                y_all.append(fut_target)

        x_hist_all = np.asarray(x_hist_all, dtype=float)
        x_future_cov_all = np.asarray(x_future_cov_all, dtype=float)
        y_all = np.asarray(y_all, dtype=float)
        if len(x_hist_all) < 2:
            raise ValueError("Too few samples for LSTM")

        scaler_hist = StandardScaler()
        scaler_future_cov = StandardScaler()
        scaler_y = StandardScaler()

        x_hist_s = scaler_hist.fit_transform(x_hist_all.reshape(-1, 2)).reshape(x_hist_all.shape)
        x_future_cov_s = scaler_future_cov.fit_transform(x_future_cov_all)
        y_s = scaler_y.fit_transform(y_all.reshape(-1, 1)).reshape(y_all.shape)

        x_hist_t = torch.tensor(x_hist_s, dtype=torch.float32, device=device)
        x_future_cov_t = torch.tensor(x_future_cov_s, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_s, dtype=torch.float32, device=device)

        ds = TensorDataset(x_hist_t, x_future_cov_t, y_t)
        dl = DataLoader(ds, batch_size=min(lstm_batch_size, len(ds)), shuffle=True, drop_last=False)

        model = LSTMQuantileForecaster(
            hidden_dim=lstm_hidden_dim,
            horizon=forecast_horizon,
            n_quantiles=len(QUANTILES),
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lstm_lr)
        loss_fn = QuantileLoss(QUANTILES)

        model.train()
        for _ in range(lstm_epochs):
            for xb_hist, xb_fc, yb in dl:
                opt.zero_grad()
                pred = model(xb_hist, xb_fc, use_bn=True)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        hist_now = np.column_stack([y_hist[-lookback:], x_hist[-lookback:]])
        fut_cov_now = np.asarray(x_future_proxy, dtype=float).reshape(1, -1)
        hist_now_s = scaler_hist.transform(hist_now).reshape(1, lookback, 2)
        fut_cov_now_s = scaler_future_cov.transform(fut_cov_now)

        hist_now_t = torch.tensor(hist_now_s, dtype=torch.float32, device=device)
        fut_cov_now_t = torch.tensor(fut_cov_now_s, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            q_scaled = model(hist_now_t, fut_cov_now_t, use_bn=False).squeeze(0).cpu().numpy()

        q = np.zeros_like(q_scaled)
        for q_idx in range(q.shape[1]):
            q[:, q_idx] = scaler_y.inverse_transform(q_scaled[:, q_idx].reshape(-1, 1)).ravel()
        return ensure_monotonic_quantiles(np.maximum(q, 0.0))

    methods = ["mantis", "ets_x", "sarimax_x", "lstm_q"]
    stats_store = {
        m: {
            "model_abs": [],
            "baseline_abs": [],
            "cov90_hits": 0,
            "cov50_hits": 0,
            "total_points": 0,
            "fallbacks": 0,
        }
        for m in methods
    }

    print(f"Evaluating file: {input_csv.name}")
    print(f"Provinces in file: {len(provinces)}")
    print(f"Using covariate column: {args.covariate_col}")

    skipped_no_covariate = 0
    skipped_too_short = 0

    for province in tqdm(provinces, desc="Provinces"):
        dfp_all = df[df["Location"] == province].sort_values("Date")

        # Explicitly skip provinces with no usable covariate values.
        cov_series = pd.to_numeric(dfp_all[args.covariate_col], errors="coerce")
        if cov_series.notna().sum() == 0:
            skipped_no_covariate += 1
            continue

        dfp = dfp_all[[args.target_col, args.covariate_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(dfp) < start_week + forecast_horizon + 1:
            skipped_too_short += 1
            continue

        y = dfp[args.target_col].astype(float).values
        x = dfp[args.covariate_col].astype(float).values

        for t in tqdm(
            range(start_week, len(y) - forecast_horizon + 1, stride),
            desc=f"{province}",
            leave=False,
        ):
            y_hist = y[:t]
            x_hist = x[:t]
            y_true = y[t:t + forecast_horizon]
            if len(y_true) < forecast_horizon:
                continue

            baseline_pred = np.repeat(y_hist[-1], forecast_horizon)
            x_future_proxy = np.repeat(x_hist[-1], forecast_horizon)

            try:
                q_mantis = forecast_mantis(y_hist, x_hist)
            except Exception:
                stats_store["mantis"]["fallbacks"] += 1
                q_mantis = np.tile(baseline_pred.reshape(-1, 1), (1, len(QUANTILES)))
            m_abs, b_abs, c90, c50 = compute_window_metrics(y_true, q_mantis, baseline_pred)
            stats_store["mantis"]["model_abs"].extend(m_abs)
            stats_store["mantis"]["baseline_abs"].extend(b_abs)
            stats_store["mantis"]["cov90_hits"] += int(c90)
            stats_store["mantis"]["cov50_hits"] += int(c50)
            stats_store["mantis"]["total_points"] += forecast_horizon

            try:
                q_ets = forecast_ets_with_covariate(y_hist, x_hist, x_future_proxy)
            except Exception:
                stats_store["ets_x"]["fallbacks"] += 1
                q_ets = np.tile(baseline_pred.reshape(-1, 1), (1, len(QUANTILES)))
            m_abs, b_abs, c90, c50 = compute_window_metrics(y_true, q_ets, baseline_pred)
            stats_store["ets_x"]["model_abs"].extend(m_abs)
            stats_store["ets_x"]["baseline_abs"].extend(b_abs)
            stats_store["ets_x"]["cov90_hits"] += int(c90)
            stats_store["ets_x"]["cov50_hits"] += int(c50)
            stats_store["ets_x"]["total_points"] += forecast_horizon

            try:
                q_sarima = forecast_sarimax_with_covariate(y_hist, x_hist, x_future_proxy)
            except Exception:
                stats_store["sarimax_x"]["fallbacks"] += 1
                q_sarima = np.tile(baseline_pred.reshape(-1, 1), (1, len(QUANTILES)))
            m_abs, b_abs, c90, c50 = compute_window_metrics(y_true, q_sarima, baseline_pred)
            stats_store["sarimax_x"]["model_abs"].extend(m_abs)
            stats_store["sarimax_x"]["baseline_abs"].extend(b_abs)
            stats_store["sarimax_x"]["cov90_hits"] += int(c90)
            stats_store["sarimax_x"]["cov50_hits"] += int(c50)
            stats_store["sarimax_x"]["total_points"] += forecast_horizon

            try:
                q_lstm = forecast_lstm_quantile(y_hist, x_hist, x_future_proxy)
            except Exception:
                stats_store["lstm_q"]["fallbacks"] += 1
                q_lstm = np.tile(baseline_pred.reshape(-1, 1), (1, len(QUANTILES)))
            m_abs, b_abs, c90, c50 = compute_window_metrics(y_true, q_lstm, baseline_pred)
            stats_store["lstm_q"]["model_abs"].extend(m_abs)
            stats_store["lstm_q"]["baseline_abs"].extend(b_abs)
            stats_store["lstm_q"]["cov90_hits"] += int(c90)
            stats_store["lstm_q"]["cov50_hits"] += int(c50)
            stats_store["lstm_q"]["total_points"] += forecast_horizon

    summary_rows = []
    mantis_errors = stats_store["mantis"]["model_abs"]
    for method in methods:
        s = stats_store[method]
        model_mae = float(np.mean(s["model_abs"])) if s["model_abs"] else np.nan
        baseline_mae = float(np.mean(s["baseline_abs"])) if s["baseline_abs"] else np.nan
        relative_mae = (
            model_mae / baseline_mae
            if (np.isfinite(baseline_mae) and baseline_mae != 0)
            else np.nan
        )
        cov90 = (s["cov90_hits"] / s["total_points"]) if s["total_points"] > 0 else np.nan
        cov50 = (s["cov50_hits"] / s["total_points"]) if s["total_points"] > 0 else np.nan
        dm_pval = np.nan if method == "mantis" else dm_test(mantis_errors, s["model_abs"], h=forecast_horizon)
        summary_rows.append(
            {
                "method": method,
                "model_mae": model_mae,
                "relative_mae": relative_mae,
                "coverage_50": cov50,
                "coverage_90": cov90,
                "dm_p_value": dm_pval,
                "fallback_count": s["fallbacks"],
                "forecast_points": s["total_points"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("relative_mae")
    print("\n=== Multimodel Probabilistic Rolling Evaluation ===")
    print(
        f"Target: {args.target_col}, Covariate: {args.covariate_col}, "
        f"Horizon: {forecast_horizon}, Stride: {stride}"
    )
    print(
        f"Skipped provinces: no covariate={skipped_no_covariate}, "
        f"too short after filtering={skipped_too_short}"
    )
    print("Relative MAE = model MAE / naive baseline MAE (lower is better; <1 beats baseline)")
    print("dm_p_value = Diebold-Mariano test p-value comparing Mantis vs. method (H0: equal predictive accuracy)")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.output_summary_csv:
        out_path = Path(args.output_summary_csv)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        summary_df.to_csv(out_path, index=False)
        print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
