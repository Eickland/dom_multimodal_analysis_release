# Multimodal dissolved organic matter analysis

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

## Refit the final models

```console
python src/run_models.py --repeats 30 --bootstrap 2000
```

The models are L2-regularized logistic regressions with balanced class weights. Preprocessing is
fitted inside each training fold. Scheme A selects the regularization parameter by grouped inner
cross-validation. Scheme B uses conservative geographic blocking and a prespecified `C=1`.

See `METHODS.md` for processing details and `LIMITATIONS.md` for unavailable source material.
