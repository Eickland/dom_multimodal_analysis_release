"""Fluorescence descriptors used by the final analysis pipelines."""

import numpy as np
import scipy.interpolate


def where_is_raman(excitation_wavelength, omega=3430.0):
    """Return the Raman Stokes-band center for an excitation wavelength in nm."""
    return 1e7 / (1e7 / excitation_wavelength - omega)


def cut_peak_spline(emission_wavelengths, fluorescence, peak_position,
                    peak_half_width=15.0, points_in_spline=10):
    """Return the quadratic spline used to replace a specified peak region."""
    mask = (
        ((emission_wavelengths < peak_position - peak_half_width)
         | (emission_wavelengths > peak_position + peak_half_width))
        & (~np.isnan(fluorescence))
    )
    knots = emission_wavelengths[mask][1:-1][::points_in_spline]
    spline = scipy.interpolate.LSQUnivariateSpline(
        emission_wavelengths[mask], fluorescence[mask], knots, k=2, ext=1
    )
    return knots, mask, emission_wavelengths[mask], spline


def cut_raman_spline(emission_wavelengths, fluorescence, excitation_wavelength,
                     raman_frequency=3430.0, raman_half_width=15,
                     points_in_spline=10):
    """Replace the Raman band with the project-standard quadratic spline."""
    result = cut_peak_spline(
        emission_wavelengths,
        fluorescence,
        peak_position=where_is_raman(excitation_wavelength, omega=raman_frequency),
        peak_half_width=raman_half_width,
        points_in_spline=points_in_spline,
    )
    return np.asarray(result[-1](emission_wavelengths))


def _nearest(values, target):
    array = np.asarray(values, dtype=float)
    return values[int(np.argmin(np.abs(array - target)))]


def calc_asm_350(data):
    """Return the 420-460/550-600 nm area ratio at 350 nm excitation."""
    emission = data.index.to_numpy(dtype=int)
    spline = cut_raman_spline(emission, data[350].to_numpy(), 350)
    high = np.trapezoid(spline[np.where(emission == 420)[0][0]:np.where(emission == 460)[0][0]])
    low = np.trapezoid(spline[np.where(emission == 550)[0][0]:np.where(emission == 600)[0][0]])
    return float(high / low)


def calc_asm_280(data):
    """Return the 350-400/475-535 nm area ratio at 280 nm excitation."""
    emission = data.index.to_numpy(dtype=int)
    spline = cut_raman_spline(emission, data[280].to_numpy(), 280)
    high = np.trapezoid(spline[np.where(emission == 350)[0][0]:np.where(emission == 400)[0][0]])
    low = np.trapezoid(spline[np.where(emission == 475)[0][0]:np.where(emission == 535)[0][0]])
    return float(high / low)


def calc_fluo_index(data):
    """Return I(450)/I(500) at the nearest 370 nm excitation."""
    row = data[_nearest(data.columns, 370)]
    return float(row.loc[_nearest(data.index, 450)] / row.loc[_nearest(data.index, 500)])


def calc_humin_index(data):
    """Return the 435-480/300-345 nm area ratio near 255 nm excitation."""
    row = data[_nearest(data.columns, 255)].to_numpy()
    emission = data.index.to_numpy(dtype=int)
    spline = cut_raman_spline(emission, row, 255)
    high = np.trapezoid(spline[np.where(emission == 435)[0][0]:np.where(emission == 480)[0][0]])
    low = np.trapezoid(spline[np.where(emission == 300)[0][0]:np.where(emission == 345)[0][0]])
    return float(high / low)
