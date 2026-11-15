"""Van Krevelen grid descriptor used by the final analysis pipelines."""

import pandas as pd


def get_squares_vk(frame, how_average="weight", ax=None, draw=False, hc_bins=None, oc_bins=None, square_indices="row_major"):
    """Return formula-count or intensity fractions in the project Van Krevelen grid."""
    data = frame.copy(deep=True)
    if "H/C" not in data or "O/C" not in data:
        data["H/C"] = data["H"] / data["C"]
        data["O/C"] = data["O"] / data["C"]
    hc_bins = hc_bins or [(1.8, 2.2), (1.4, 1.8), (1.0, 1.4), (0.6, 1.0), (0.0, 0.6)]
    oc_bins = oc_bins or [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    denominator = data["intensity"].sum() if how_average == "weight" else len(data)
    standard = {
        (0, 0): 5, (0, 1): 10, (0, 2): 15, (0, 3): 20,
        (1, 0): 4, (1, 1): 9, (1, 2): 14, (1, 3): 19,
        (2, 0): 3, (2, 1): 8, (2, 2): 13, (2, 3): 18,
        (3, 0): 2, (3, 1): 7, (3, 2): 12, (3, 3): 17,
        (4, 0): 1, (4, 1): 6, (4, 2): 11, (4, 3): 16,
    }
    rows = []
    for row, (hc_min, hc_max) in enumerate(hc_bins):
        for col, (oc_min, oc_max) in enumerate(oc_bins):
            selected = data[
                data["H/C"].ge(hc_min) & data["H/C"].lt(hc_max)
                & data["O/C"].ge(oc_min) & data["O/C"].lt(oc_max)
            ]
            if how_average == "count":
                value = len(selected) / denominator
            elif how_average == "weight":
                value = selected["intensity"].sum() / denominator
            else:
                raise ValueError("how_average must be 'count' or 'weight'")
            if square_indices == "standard":
                square = standard[(row, col)]
            elif square_indices == "col_major":
                square = col * len(hc_bins) + row + 1
            else:
                square = row * len(oc_bins) + col + 1
            rows.append({
                "value": value, "square_index": square, "row": row, "col": col,
                "hc_range": f"{hc_min}-{hc_max}", "oc_range": f"{oc_min}-{oc_max}",
                "hc_min": hc_min, "hc_max": hc_max, "oc_min": oc_min, "oc_max": oc_max,
            })
    return pd.DataFrame(rows).sort_values("square_index").reset_index(drop=True)
