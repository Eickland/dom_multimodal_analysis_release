# Data processing and analysis

## UV-Vis and fluorescence

UV-Vis files are instrument-baseline-corrected acquisition exports. A documented legacy export
anomaly, in which a small number of absorbance values above 1 were stored as three-decimal
integers, is repaired by dividing values with absolute magnitude above 10 by 1000. No retrospective
baseline subtraction is applied.

Fluorescence EEMs are corrected for inner-filter effects using decadic absorbance in the same 1 cm
geometry:

`F_corrected(Ex, Em) = F_observed(Ex, Em) * 10 ** ((A(Ex) + A(Em)) / 2)`

The public optical modules contain the UV interpolation, EEM correction, Raman-band spline, and
descriptor calculations used for the final optical feature table.

## FT-ICR MS

The raw-data directory contains the current instrument assignment exports, not profile spectra.
The processed directory contains one final molecular-formula table per sample, calibrated peak
tables, calibration statistics, solvent-blank formulas, and sample-level removal counts.

Formula selection used mass-error and chemical-plausibility criteria. Duplicate assignments were
resolved deterministically. Solvent-blank formulas, the predefined negative-ESI contaminant list,
and polyether-like C2H4O/C3H6O series were excluded. Final formula tables additionally apply the
documented H/C, O/C, oxygen-count, elemental-count, and error constraints. The exact per-sample
outcomes are recorded in `summary_qc.csv` and `removed_formulas_per_sample.csv`.

## Models

The final cohort contains 59 samples. The public model script reads the frozen optical and
molecular feature matrices and the exact outer split table. Model families are DOC-only, optical,
molecular, and combined. The main optical sensitivity analysis excludes decomposition-derived
features. All imputation, constant-feature filtering, scaling, DOC normalization, and residualization
are fitted using training data only.

The fixed random seed is 429. Reported predictions are out-of-fold probabilities averaged across
repeated grouped outer cross-validation.
