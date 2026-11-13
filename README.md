© 2026 The Regents of the University of Michigan

Carson Dudley -- University of Michigan

---

# Mantis 2: A Foundation Model for Infectious Disease Forecasting

![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm--Noncommercial%201.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Downloads](https://img.shields.io/github/downloads/carsondudley1/Mantis/total)


Mantis 2 is a probabilistic forecasting model trained on simulated outbreaks and
real-world disease surveillance data. It forecasts up to six weekly time series
jointly, returning 21 quantiles for each of the next 16 weeks.

---

## Try It

[Run the Colab Tutorial](https://colab.research.google.com/drive/1Epuq-6ZGUM67FOfWHnLGkld4-cb8EDW0?usp=sharing)  
_No installation or coding required._

---

## Paper

- **Mantis Paper:** [Mantis: A Foundation Model for Mechanistic Disease Forecasting](https://www.pnas.org/doi/10.1073/pnas.2602542123) <br>
  *Carson Dudley et al., PNAS*
---

## Installation

Mantis 2 requires Python 3.10 or later.

```bash
python -m pip install \
  "mantis[v2] @ git+https://github.com/carsondudley1/Mantis.git@mantis-v2.0.0"
```

## Model checkpoint

Download `config.json` and `model.safetensors` from the
[Mantis 2 release](https://github.com/carsondudley1/Mantis/releases/tag/mantis-v2.0.0)
and place both files in `models/mantis2/`. You can also download them from a terminal.

```bash
mkdir -p models/mantis2
release_url="https://github.com/carsondudley1/Mantis/releases/download/mantis-v2.0.0"

curl -fL "$release_url/config.json" -o models/mantis2/config.json
curl -fL "$release_url/model.safetensors" -o models/mantis2/model.safetensors
```

## Quick Example

```
from mantis.v2 import Mantis2

# Load the model
model = Mantis2()

# Predict on example data
forecast = model.predict(time_series=deaths_ts)

# Mantis 2 will forecast the time series (dimension = 1) with 21 quantiles and for 16 weeks
print(forecast.shape)  # (1, 21, 16)
```

`deaths_ts` is a sequence of at least eight weekly observations in the original
units, ordered oldest to newest. Use `NaN` for missing weeks. Mantis 2 generates
seasonal features internally using a fixed starting phase.

To forecast several aligned series together, pass one row per series:

```python
forecast = model.predict(time_series=[deaths_ts, hospitalizations_ts])
```

See full usage in mantis_forecasting_demo.ipynb

## Project Structure

``` text
mantis/ # Core library
├── model.py # Model architecture
├── inference.py # High-level Mantis class
├── utils.py # Preprocessing & normalization
├── __init__.py

mantis_forecasting_demo.ipynb # Main Colab demo
data/ # Example input data (CSV)
models/ # Model weights (downloaded from Releases)
setup.py
LICENSE.txt
NOTICE.txt
```

## Citation

If you use Mantis in academic work, please cite the Mantis paper:

```bibtex
@article{mantis,
  title     = {Mantis: A foundation model for mechanistic disease forecasting},
  author    = {Dudley, Carson and Magdaleno, Reiden and Harding, Christopher and Eisenberg, Marisa},
  journal   = {Proceedings of the National Academy of Sciences},
  volume    = {123},
  number    = {40},
  pages     = {e2602542123},
  year      = {2026},
  publisher = {National Academy of Sciences},
  doi       = {10.1073/pnas.2602542123},
  url       = {https://doi.org/10.1073/pnas.2602542123}
}
```

