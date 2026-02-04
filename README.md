© 2025 The Regents of the University of Michigan  
Carson Dudley -- University of Michigan

---

# 🦠 Mantis: A Foundation Model for Infectious Disease Forecasting

![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm--Noncommercial%201.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Downloads](https://img.shields.io/github/downloads/carsondudley1/Mantis/total)


**Mantis** is a foundation model trained entirely on mechanistic outbreak simulations.  
It forecasts infectious disease trajectories **without needing any real-world training data**, and generalizes across diseases, regions, and outbreak settings.

---

## 🚀 Try It

👉 [Run the Colab Tutorial](https://colab.research.google.com/drive/1Epuq-6ZGUM67FOfWHnLGkld4-cb8EDW0?usp=sharing)  
_No installation or coding required._

---

## 📚 Paper

- **Mantis Paper:** [Mantis: A Simulation-Grounded Foundation Model for Disease Forecasting](https://arxiv.org/abs/2508.12260) <br>
  *Carson Dudley et al., arXiv:2508.12260 [cs.AI]*
---

## 📦 Installation

To install the latest version directly from GitHub:

```bash
pip install git+https://github.com/carsondudley1/Mantis.git
```

## 💾 Model Weights

Model weights are available on the [Releases page](https://github.com/carsondudley1/Mantis/releases).  
Each `.pt` file corresponds to a specific configuration (4-week / 8-week, with / without covariates).  

After installation, download your desired model like this:

```bash
mkdir -p models
wget -O models/mantis_4w_cov.pt https://github.com/carsondudley1/Mantis/releases/download/mantis-v1.0/mantis_4w_cov.pt
```

## 🧪 Quick Example

```
from mantis import Mantis

# Load a 4-week horizon model with covariates
model = Mantis(forecast_horizon=4, use_covariate=True)

# Predict on example data
forecast = model.predict(
    time_series=deaths_ts,         # e.g. past deaths
    covariate=hosp_ts,         # e.g. past hospitalizations
    target_type=2,                # 0 = cases, 1 = hosp, 2 = deaths
    covariate_type=1
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
models/ # Model weights (downloaded from Releases)
setup.py
LICENSE.txt
NOTICE.txt
```

## 📄 Citation

If you use Mantis in academic work, please cite the Mantis paper:

```bibtex
@article{mantis,
  title={Mantis: A Foundation Model for Mechanistic Disease Forecasting},
  author={Carson Dudley and Reiden Magdaleno and Christopher Harding and Ananya Sharma and Emily Martin and Marisa Eisenberg},
  journal={arXiv preprint arXiv:2508.12260},
  year={2025},
  url={https://arxiv.org/abs/2508.12260}
}
```

