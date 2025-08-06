© 2025 The Regents of the University of Michigan  
Carson Dudley — University of Michigan

---

# 🦠 Mantis: A Simulation-Grounded Foundation Model for Infectious Disease Forecasting

![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm--Noncommercial%201.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Downloads](https://img.shields.io/github/downloads/carsondudley1/Mantis/total)


**Mantis** is a foundation model trained entirely on mechanistic outbreak simulations.  
It forecasts infectious disease trajectories **without needing any real-world training data**, and generalizes across diseases, regions, and outbreak settings.

> A simulation-trained, zero-shot infectious disease forecasting model with mechanistic interpretability.

---

## 🚀 Try It Instantly

👉 [Run the Colab Tutorial](https://colab.research.google.com/drive/1Epuq-6ZGUM67FOfWHnLGkld4-cb8EDW0?usp=sharing)  
_No installation or coding required._

---

## 📚 Papers

- **Mantis Paper:** _Coming soon — link will be here_
- **SGNN Framework Paper:** [Simulation as Supervision: Mechanistic Pretraining for Scientific Discovery](https://arxiv.org/abs/2507.08977)  
  *Carson Dudley et al., arXiv:2507.08977 [cs.LG]*
---

## 📦 Installation

To install the latest version directly from GitHub:

```bash
pip install git+https://github.com/carsondudley1E/Mantis.git
```

## 💾 Model Weights

Model weights are available on the [Releases page](https://github.com/carsondudley1/Mantis/releases).  
Each `.pt` file corresponds to a specific configuration (4-week / 8-week, with / without covariates).  

After installation, download your desired model like this:

```bash
mkdir -p models
wget -O models/mantis_4w_cov.pt https://github.com/YOUR_USERNAME/Mantis/releases/download/mantis-v1.0/mantis_4w_cov.pt
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
📘 See full usage in mantis_forecasting_demo.ipynb

## 📁 Project Structure

``` text
mantis/ # Core library
├── model.py # Model architecture
├── inference.py # High-level Mantis class
├── utils.py # Preprocessing & normalization
├── __init__.py

mantis_forecasting_demo.ipynb # ✅ Main Colab demo
data/ # Example input data (CSV)
models/ # Model weights (optional; downloaded from Releases)
setup.py
LICENSE.txt
NOTICE.txt
```

## 📄 Citation

If you use Mantis or the SGNN framework in academic work, please cite the Mantis and SGNN papers:

```bibtex
@misc{sgnns2025,
  title     = {Simulation as Supervision: Mechanistic Pretraining for Scientific Discovery},
  author    = {Carson Dudley and Reiden Magdaleno and Christopher Harding and Marisa Eisenberg},
  year      = {2025},
  eprint    = {2507.08977},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2507.08977}
}
```
