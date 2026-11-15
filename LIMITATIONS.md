# Limitations of the public release

The profile mzML files were stored outside the available workspace and are not included. Two source
files named in the FT-ICR MS processing notes (`scripts/final_pipeline.py` and
`scripts/summarize.py`) were also absent. Therefore, calibration and formula assignment cannot be
reproduced from profile spectra using this repository alone. The instrument assignment exports,
final peaks, formulas, calibration statistics, and contaminant-removal logs are included so that
the available processing trail remains auditable.

No serialized estimator objects were present. Models are refitted from the included feature tables,
cross-validation splits, and code.

Geographic coordinates were not required for the released analysis and are intentionally omitted.
