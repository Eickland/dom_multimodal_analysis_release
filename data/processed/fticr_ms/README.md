# Final FT-ICR MS spectra

This directory contains 64 sample-level final formula lists and 65 calibrated peak tables (the 64 samples plus the solvent blank `BLANK_MeOH`). File names carry the instrument export suffix (e.g. `ADOM-SL1-10_000001.csv`). The profile mzML files used for peak detection are not present in the source workspace and are therefore not included in this release. The instrument assignment exports are included as available source records in `data/raw/fticr_ms_exports/`.

Five of the 64 samples are not part of the 59-sample modelling cohort (`data/processed/model_inputs/sample_registry.csv`): ADOM-SL5-1, ADOM-SL5-3 and ADOM-SL7-4 (no UV-Vis or EEM export), ADOM-SL9-6 (no UV-Vis export) and ADOM-SLB-B3-2 (sampled at a geographic location different from all other samples). Their formula tables are released for completeness.

## Files

- `formulas/<sample>.csv`: final molecular formulas and descriptors.
- `peaks/<sample>.csv`: detected peaks with calibrated m/z, assignment, isotope, contaminant, polyether-series and final-selection flags.
- `all_formulas.csv`: combined formula table.
- `summary_descriptors.csv`: sample-level formula descriptors.
- `summary_qc.csv`: calibration and assignment quality metrics.
- `removed_formulas_per_sample.csv`: blank, contaminant-list and polyether removals.
- `blank_formulas.csv`: solvent-blank formulas.

## Processing summary

Peaks were detected at local S/N >= 6 using 50 Da local-noise windows and iterative median/MAD clipping. Centroids were estimated from the three highest profile points. Each sample was recalibrated using CHO calibrants supported by plausible 13C isotopologues, with a robust LOWESS error trend and an optional periodic 1/m correction selected by five-fold cross-validation. Formula assignment used [M-H]- ions, C4-50, H4-100, O0-25, N0-1 and S0-1, chemical plausibility filters, and a sample-specific tolerance bounded to 0.5-1.0 ppm. Duplicate formula assignments retained the more intense peak.

Solvent-blank formulas, a predefined negative-ESI contaminant list, and polyether-like C2H4O/C3H6O series were flagged and excluded from the final formula set. Final formulas also required 0.3 <= H/C <= 2.0, O/C < 1 and O >= 1. Detailed removal counts and calibration statistics are provided in the two summary logs.

The source workspace documentation referred to `scripts/final_pipeline.py` and `scripts/summarize.py`, but those two files were not present when this release was assembled. This limitation is recorded in the root `LIMITATIONS.md`.
