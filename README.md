© 2025 The Regents of the University of Michigan  
Carson Dudley — University of Michigan

---

# 🦠 Mantis: A Simulation-Grounded Foundation Model for Infectious Disease Forecasting

**Mantis** is a foundation model trained entirely on mechanistic outbreak simulations.  
It forecasts infectious disease trajectories **without needing any real-world training data**, and generalizes across diseases, regions, and outbreak settings.

---

## 🚀 Try It Instantly

👉 [Run the Colab Tutorial](https://colab.research.google.com/drive/1Epuq-6ZGUM67FOfWHnLGkld4-cb8EDW0?usp=sharing)  
_No installation or coding required._

---

## 📚 Papers

- **Mantis Paper:** _Coming soon — link will be here_
- **SGNN Framework Paper:** 

---

## 🔍 Why Mantis?

- 📦 **Zero-shot forecasting**: No fine-tuning or retraining needed
- 🌎 **Cross-disease generalization**: Trained on simulated data covering broad pathogen space
- 🧠 **Mechanistic interpretability**: Forecasts backed by simulation-driven reasoning
- ⚙️ **Flexible & fast**: Predicts cases, hospitalizations, or deaths from partial signals

---

## 📦 Installation

To install the latest version directly from GitHub:

```bash
pip install git+https://github.com/YOUR_USERNAME/Mantis.git
```

## 🧪 Quick Example

```
from mantis import Mantis

# Load a 4-week horizon model with covariates
model = Mantis(forecast_horizon=4, use_covariate=True)

# Predict on example data
forecast = model.predict(
    time_series=hosp_ts,         # e.g. past hospitalizations
    covariate=deaths_ts,         # e.g. past deaths
    population=1_000_000,
    target_type=1                # 0 = cases, 1 = hosp, 2 = deaths
)
```
