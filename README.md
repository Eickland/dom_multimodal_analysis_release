# Optical and FT-ICR MS signatures of dissolved organic matter from sludge-lignin at Lake Baikal

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24259061.svg)](https://doi.org/10.5281/zenodo.24259061)

Code and data accompanying the article:

> K. S. Larionov, K. V. Petrov, A. N. Khreptugova, M. G. Zavyalova, E. N. Nikolaev, R. K. Fedorov and I. V. Perminova. Optical and FT-ICR MS signatures of dissolved organic matter from sludge-lignin at Lake Baikal. *Analytical Methods*, 2026 (submitted). DOI: to be added on publication.

Correspondence: K. S. Larionov, larionovks@my.msu.ru

This public repository contains the primary data available for the study, the final processed
FT-ICR MS tables, the model-ready optical and molecular descriptors, and the main code needed to
refit the reported grouped logistic-regression models.

## Structure

- `data/raw/`: UV-Vis exports, fluorescence EEM exports, organic-carbon measurements, and the
  current FT-ICR MS assignment exports.
- `data/processed/fticr_ms/`: calibrated peak tables, molecular-formula tables, calibration QC,
  formula-removal logs, blank formulas, and final descriptors.
- `data/processed/model_inputs/`: the frozen 59-sample cohort, optical descriptors, molecular
  features, model feature definitions, and outer cross-validation splits.
- `src/`: compact public code for optical corrections and descriptors, molecular descriptors, and
  the final grouped models.
- `results/`: only the principal model tables and aggregated out-of-fold predictions.
- `manifests/`: SHA-256 checksums and source-code provenance.

## Installation

The analysis was run with Python 3.12 (see `environment/python.txt`). Install the pinned
dependencies from the repository root:

```console
pip install -r environment/requirements.txt
```

## Refit the final models

Run from the repository root:

```console
python src/run_models.py --repeats 30 --bootstrap 2000
```

The models are L2-regularized logistic regressions with balanced class weights. Preprocessing is
fitted inside each training fold. Scheme A selects the regularization parameter by grouped inner
cross-validation. Scheme B uses conservative geographic blocking and a prespecified `C=1`.

Outputs are written to `results/recomputed/` (override with `--output`). Compare
`results/recomputed/model_performance.csv` and `oof_predictions_aggregated.csv` with the released
tables in `results/`.

See `METHODS.md` for processing details and `LIMITATIONS.md` for unavailable source material.

## How to cite

Please cite the article above and this archive (see `CITATION.cff`):

> K. S. Larionov, K. V. Petrov, A. N. Khreptugova, M. G. Zavyalova, E. N. Nikolaev, R. K. Fedorov and I. V. Perminova. Optical and FT-ICR MS signatures of dissolved organic matter from sludge-lignin at Lake Baikal (code and data), version 1.0.2. Zenodo, 2026. https://doi.org/10.5281/zenodo.24259061

## License

Code and data in this repository are released under the MIT License (see `LICENSE`).
