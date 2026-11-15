"""Absorbance-based inner-filter correction for the final EEM workflow.

The implementation follows the equation specified in the analysis protocol:

    F_corrected(Ex, Em) = F_observed(Ex, Em) * 10 ** ((A(Ex) + A(Em)) / 2)

Absorbance is decadic absorbance measured in the same 1-cm sample geometry.
No retrospective UV baseline subtraction is performed in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IFEResult:
    corrected: np.ndarray
    correction_factor: np.ndarray
    a_total: np.ndarray
    a_excitation: np.ndarray
    a_emission: np.ndarray


def read_instrument_corrected_uv(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Read a two-column UV export without applying any baseline subtraction.

    A legacy export anomaly stored a small subset of absorbances above 1 as
    three-decimal integers (for example, 1.092 as 1092). Values whose absolute
    magnitude exceeds 10 are restored by division by 1000. The count is
    returned so the repair remains auditable.
    """

    frame = pd.read_csv(path, dtype=str)
    wavelength = pd.to_numeric(
        frame.iloc[:, 0].str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )
    absorbance = pd.to_numeric(
        frame.iloc[:, 1].str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )
    valid = wavelength.notna() & absorbance.notna()
    table = pd.DataFrame(
        {
            "wavelength_nm": wavelength.loc[valid].astype(float),
            "absorbance": absorbance.loc[valid].astype(float),
        }
    )
    table = table.groupby("wavelength_nm", as_index=False)["absorbance"].mean()
    table = table.sort_values("wavelength_nm").reset_index(drop=True)
    values = table["absorbance"].to_numpy(float)
    legacy = np.isfinite(values) & (np.abs(values) > 10)
    values = values.copy()
    values[legacy] /= 1000.0
    return table["wavelength_nm"].to_numpy(float), values, int(legacy.sum())


def interpolate_absorbance(
    wavelength_nm: np.ndarray,
    absorbance: np.ndarray,
    targets_nm: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate absorbance and reject extrapolation."""

    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    absorbance = np.asarray(absorbance, dtype=float)
    targets_nm = np.asarray(targets_nm, dtype=float)
    if wavelength_nm.ndim != 1 or absorbance.ndim != 1:
        raise ValueError("UV wavelength and absorbance must be one-dimensional")
    if wavelength_nm.size != absorbance.size or wavelength_nm.size < 2:
        raise ValueError("UV wavelength and absorbance arrays are incompatible")
    if np.any(np.diff(wavelength_nm) <= 0):
        raise ValueError("UV wavelengths must be strictly increasing")
    if np.any(~np.isfinite(wavelength_nm)) or np.any(~np.isfinite(absorbance)):
        raise ValueError("UV spectrum contains non-finite values")
    tolerance = 1e-9
    if targets_nm.min() < wavelength_nm.min() - tolerance:
        raise ValueError("IFE target wavelength is below the UV measurement range")
    if targets_nm.max() > wavelength_nm.max() + tolerance:
        raise ValueError("IFE target wavelength is above the UV measurement range")
    return np.interp(targets_nm, wavelength_nm, absorbance)


def correct_eem_aba(
    observed: np.ndarray,
    emission_nm: np.ndarray,
    excitation_nm: np.ndarray,
    uv_wavelength_nm: np.ndarray,
    uv_absorbance: np.ndarray,
    *,
    clip_negative_absorbance: bool,
) -> IFEResult:
    """Apply the Kothawala absorbance-based correction on an EEM grid."""

    observed = np.asarray(observed, dtype=float)
    emission_nm = np.asarray(emission_nm, dtype=float)
    excitation_nm = np.asarray(excitation_nm, dtype=float)
    if observed.shape != (emission_nm.size, excitation_nm.size):
        raise ValueError("EEM dimensions do not match emission/excitation axes")
    absorbance = np.asarray(uv_absorbance, dtype=float).copy()
    if clip_negative_absorbance:
        absorbance = np.maximum(absorbance, 0.0)
    a_excitation = interpolate_absorbance(uv_wavelength_nm, absorbance, excitation_nm)
    a_emission = interpolate_absorbance(uv_wavelength_nm, absorbance, emission_nm)
    a_total = a_emission[:, None] + a_excitation[None, :]
    factor = np.power(10.0, a_total / 2.0)
    corrected = observed * factor
    if corrected.shape != observed.shape:
        raise AssertionError("IFE correction changed EEM dimensions")
    return IFEResult(
        corrected=corrected,
        correction_factor=factor,
        a_total=a_total,
        a_excitation=a_excitation,
        a_emission=a_emission,
    )


def write_eem_csv(
    path: Path,
    matrix: np.ndarray,
    emission_nm: np.ndarray,
    excitation_nm: np.ndarray,
) -> None:
    """Write a machine-readable EEM table with one emission row per record."""

    frame = pd.DataFrame(
        np.asarray(matrix, dtype=float),
        columns=[f"Ex_{value:g}_nm" for value in np.asarray(excitation_nm, dtype=float)],
    )
    frame.insert(0, "emission_nm", np.asarray(emission_nm, dtype=float))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
