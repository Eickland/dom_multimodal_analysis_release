"""CDOM descriptors used by the final analysis pipelines."""

import numpy as np


def calc_ag(absorbance_series, path_length_cm=1.0):
    """Return the Napierian absorption coefficient a_g(lambda), in m^-1."""
    return 2.303 * absorbance_series / (path_length_cm / 100)


def calc_spectral_slope(ag_series, lambda1, lambda2):
    """Return the log-linear spectral slope between two wavelengths."""
    if lambda1 not in ag_series.index or lambda2 not in ag_series.index:
        ag1 = np.interp(lambda1, ag_series.index, ag_series.values)
        ag2 = np.interp(lambda2, ag_series.index, ag_series.values)
    else:
        ag1 = ag_series.loc[lambda1]
        ag2 = ag_series.loc[lambda2]
    if ag1 <= 0 or ag2 <= 0:
        return np.nan
    return -(np.log(ag1) - np.log(ag2)) / (lambda1 - lambda2)


def _band_residual(ag_series, left, center, right):
    wavelengths = ag_series.index
    values = [
        ag_series.loc[wave] if wave in wavelengths else np.interp(wave, wavelengths, ag_series.values)
        for wave in (left, center, right)
    ]
    ag_left, ag_center, ag_right = values
    slope = (np.log(ag_right) - np.log(ag_left)) / (right - left)
    expected = ag_left * np.exp(slope * (center - left))
    return ag_center - expected


def calc_B1_band(ag_series):
    return _band_residual(ag_series, 295, 305, 330)


def calc_B2_band(ag_series):
    return _band_residual(ag_series, 380, 410, 443)


def calc_cdom_descriptors(absorbance_df, path_length_cm=1.0, doc_concentration=None):
    """Calculate the CDOM descriptor dictionary used by the final models."""
    absorbance = absorbance_df.iloc[:, 0]
    ag = calc_ag(absorbance, path_length_cm)
    ag_275 = ag.loc[275] if 275 in ag.index else np.interp(275, ag.index, ag.values)
    ag_380 = ag.loc[380] if 380 in ag.index else np.interp(380, ag.index, ag.values)
    output = {
        "ag_275": ag_275,
        "ag_380": ag_380,
        "S_275_295": calc_spectral_slope(ag, 275, 295),
        "S_380_443": calc_spectral_slope(ag, 380, 443),
        "B1_prime": calc_B1_band(ag),
        "B2_prime": calc_B2_band(ag),
    }
    if doc_concentration is not None and doc_concentration > 0:
        for name in ("ag_275", "ag_380", "B1_prime", "B2_prime"):
            output[f"{name}_DOC"] = output[name] / doc_concentration
    return output
