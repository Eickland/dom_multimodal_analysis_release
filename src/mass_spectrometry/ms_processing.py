from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from nomspectra.spectrum import Spectrum

from src.mass_spectrometry import mass_descriptors as hum_ms


NATIVE_HC_BINS = [(0.0, 0.6), (0.6, 1.0), (1.0, 1.4), (1.4, 1.8), (1.8, 2.2)]
NATIVE_OC_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
MAIN_ELEMENTAL_CLASSES = ("CHO", "CHON", "CHOS", "CHONS")
PERMINOVA_CLASSES = (
    "condensed_tanins",
    "hydrolyzable_tanins",
    "phenylisopropanoids",
    "terpenoids",
    "lipids",
    "proteins",
    "carbohydrates",
)
CHEMOTYPE_CANONICAL = {
    "condensed_tanins": "condensed_tannin_like",
    "hydrolyzable_tanins": "hydrolysable_tannin_like",
    "phenylisopropanoids": "lignin_phenylpropanoid_like",
    "terpenoids": "terpenoid_like",
    "lipids": "lipid_like",
    "proteins": "peptide_protein_like",
    "carbohydrates": "carbohydrate_like",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt_list(values: Iterable[Any]) -> str:
    output = []
    for value in values:
        if pd.isna(value):
            output.append("")
        elif isinstance(value, (float, np.floating)):
            output.append(f"{float(value):.15g}")
        else:
            output.append(str(value))
    return "|".join(output)


def _tie_rule(group: pd.DataFrame) -> str:
    candidates = group
    minimum = candidates["_abs_error_ppm"].min()
    candidates = candidates.loc[candidates["_abs_error_ppm"].eq(minimum)]
    if len(candidates) == 1:
        return "minimum_absolute_assignment_error_ppm"
    maximum_intensity = candidates["intensity"].max()
    candidates = candidates.loc[candidates["intensity"].eq(maximum_intensity)]
    if len(candidates) == 1:
        return "ppm_tie_then_maximum_intensity"
    minimum_da = candidates["_abs_error_da"].min()
    candidates = candidates.loc[candidates["_abs_error_da"].eq(minimum_da)]
    if len(candidates) == 1:
        return "ppm_and_intensity_tie_then_minimum_absolute_error_da"
    minimum_mz = candidates["mass"].min()
    candidates = candidates.loc[candidates["mass"].eq(minimum_mz)]
    if len(candidates) == 1:
        return "ppm_intensity_da_tie_then_minimum_mz"
    return "complete_numeric_tie_then_minimum_original_row"


def resolve_duplicate_formulas(
    source: pd.DataFrame,
    sample_id: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Keep one assigned row per exact formula without summing intensities."""

    formula_column = str(config["formula_column"])
    assign_column = str(config["assignment_flag_column"])
    ppm_column = str(config["assignment_error_ppm_column"])
    da_column = str(config["assignment_error_da_column"])
    required = {
        formula_column,
        assign_column,
        ppm_column,
        da_column,
        "mass",
        "intensity",
        "calc_mass",
        "C",
        "H",
        "O",
        "N",
        "S",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{sample_id}: missing required mass-list columns {missing}")
    if ppm_column != "rel_error" or da_column != "abs_error":
        raise ValueError(
            f"{sample_id}: configured error columns must be the audited source columns "
            "rel_error (ppm) and abs_error (Da)"
        )

    table = source.copy()
    table["original_row"] = np.arange(1, len(table) + 1, dtype=int)
    assigned = table.loc[table[assign_column].astype(bool)].copy()
    if assigned[formula_column].isna().any():
        raise ValueError(f"{sample_id}: assigned rows contain missing formulas")
    if (~np.isfinite(pd.to_numeric(assigned[ppm_column], errors="coerce"))).any():
        raise ValueError(f"{sample_id}: assignment-error ppm column contains non-finite values")
    if (assigned["intensity"] < 0).any():
        raise ValueError(f"{sample_id}: assigned rows contain negative intensities")

    assigned["_abs_error_ppm"] = assigned[ppm_column].astype(float).abs()
    assigned["_abs_error_da"] = assigned[da_column].astype(float).abs()
    ordered = assigned.sort_values(
        [formula_column, "_abs_error_ppm", "intensity", "_abs_error_da", "mass", "original_row"],
        ascending=[True, True, False, True, True, True],
        kind="mergesort",
    )
    retained = ordered.drop_duplicates(formula_column, keep="first").copy()
    retained = retained.sort_values("original_row", kind="mergesort").reset_index(drop=True)
    retained["selection_error_column"] = ppm_column
    retained["selection_error_units"] = "ppm"
    retained["retained_absolute_error_ppm"] = retained["_abs_error_ppm"]

    duplicate_mask = assigned.duplicated(formula_column, keep=False)
    duplicate_rows = assigned.loc[duplicate_mask].copy()
    retained_index = retained.set_index(formula_column)
    audit_rows: list[dict[str, Any]] = []
    for formula, group in duplicate_rows.groupby(formula_column, sort=True):
        selected = retained_index.loc[formula]
        rejected = group.loc[group["original_row"].ne(int(selected["original_row"]))].sort_values(
            "original_row"
        )
        audit_rows.append(
            {
                "sample_id": sample_id,
                "molecular_formula": formula,
                "duplicate_count": int(len(group)),
                "retained_original_row": int(selected["original_row"]),
                "retained_mz": float(selected["mass"]),
                "retained_intensity": float(selected["intensity"]),
                "retained_error_ppm": float(selected[ppm_column]),
                "rejected_rows": _fmt_list(rejected["original_row"]),
                "rejected_mz_values": _fmt_list(rejected["mass"]),
                "rejected_intensities": _fmt_list(rejected["intensity"]),
                "rejected_error_ppm_values": _fmt_list(rejected[ppm_column]),
                "tie_break_rule_used": _tie_rule(group),
            }
        )

    group_sizes = assigned.groupby(formula_column, sort=False).size()
    summary = {
        "sample_id": sample_id,
        "raw_rows_in_file": int(len(source)),
        "assigned_rows_before_resolution": int(len(assigned)),
        "unique_formulas_after_resolution": int(len(retained)),
        "duplicate_formula_groups": int((group_sizes > 1).sum()),
        "rows_removed": int(len(assigned) - len(retained)),
        "rows_removed_percent": float(100 * (len(assigned) - len(retained)) / len(assigned)),
        "median_absolute_error_ppm_before": float(assigned["_abs_error_ppm"].median()),
        "median_absolute_error_ppm_after": float(retained["_abs_error_ppm"].median()),
        "total_intensity_before": float(assigned["intensity"].sum()),
        "total_intensity_after": float(retained["intensity"].sum()),
        "retained_to_original_intensity_ratio": float(
            retained["intensity"].sum() / assigned["intensity"].sum()
        ),
        "assignment_error_ppm_column": ppm_column,
        "assignment_error_da_column": da_column,
    }
    retained = retained.drop(columns=["_abs_error_ppm", "_abs_error_da"])
    return retained, pd.DataFrame(audit_rows), summary


def _half_open_bin(value: float, bins: list[tuple[float, float]]) -> int | None:
    for index, (lower, upper) in enumerate(bins):
        if lower <= value < upper:
            return index
    return None


def assign_vk_cell(hc: float, oc: float) -> str | None:
    hc_i = _half_open_bin(float(hc), NATIVE_HC_BINS)
    oc_i = _half_open_bin(float(oc), NATIVE_OC_BINS)
    if hc_i is None or oc_i is None:
        return None
    return f"D{oc_i * 5 + hc_i + 1:02d}"


def vk_definition_frame() -> pd.DataFrame:
    rows = []
    for oc_i, (oc_min, oc_max) in enumerate(NATIVE_OC_BINS):
        for hc_i, (hc_min, hc_max) in enumerate(NATIVE_HC_BINS):
            number = oc_i * 5 + hc_i + 1
            rows.append(
                {
                    "stable_cell_id": f"D{number:02d}",
                    "nomspectra_square": number,
                    "HC_min": hc_min,
                    "HC_max": hc_max,
                    "OC_min": oc_min,
                    "OC_max": oc_max,
                    "lower_bounds_inclusive": True,
                    "upper_bounds_inclusive": False,
                    "definition": "NOMspectra native 5x4 van Krevelen grid",
                    "source": "https://doi.org/10.1021/jasms.3c00003",
                }
            )
    return pd.DataFrame(rows)


def chemotype_definition_frame() -> pd.DataFrame:
    rows = [
        ("condensed_tannin_like", "condensed_tanins", "D01,D02,D06,D07"),
        ("lignin_phenylpropanoid_like", "phenylisopropanoids", "D03,D08"),
        ("terpenoid_like", "terpenoids", "D04,D09"),
        ("lipid_like", "lipids", "D05"),
        ("peptide_protein_like", "proteins", "D10"),
        ("hydrolysable_tannin_like", "hydrolyzable_tanins", "D11,D12,D13,D16,D17,D18"),
        ("carbohydrate_like", "carbohydrates", "D14,D15,D19,D20"),
    ]
    return pd.DataFrame(rows, columns=["chemotype", "nomspectra_class", "native_vk_cells"])


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights)
    if not valid.any() or float(weights.loc[valid].sum()) <= 0:
        return float("nan")
    return float(np.average(values.loc[valid].astype(float), weights=weights.loc[valid].astype(float)))


def _safe_ai(table: pd.DataFrame) -> pd.Series:
    numerator = table["DBE_AI"].astype(float)
    denominator = table["CAI"].astype(float)
    return (numerator / denominator).where(denominator != 0, np.nan)


def _elemental_class(table: pd.DataFrame) -> pd.Series:
    oxygenated = table["O"].astype(float).gt(0)
    nitrogen = table["N"].astype(float).gt(0)
    sulfur = table["S"].astype(float).gt(0)
    result = pd.Series("OTHER_NON_OXYGENATED", index=table.index, dtype=object)
    result.loc[oxygenated & ~nitrogen & ~sulfur] = "CHO"
    result.loc[oxygenated & nitrogen & ~sulfur] = "CHON"
    result.loc[oxygenated & ~nitrogen & sulfur] = "CHOS"
    result.loc[oxygenated & nitrogen & sulfur] = "CHONS"
    return result


def calculate_descriptors(
    retained: pd.DataFrame,
    sample_id: str,
    class_label: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    base_columns = [column for column in retained.columns if column not in {
        "original_row", "selection_error_column", "selection_error_units", "retained_absolute_error_ppm"
    }]
    spec = Spectrum(retained[base_columns].copy()).hc_oc().dbe().dbe_o().dbe_ai().cai().nosc().cram()
    table = spec.table.copy()
    table["AI"] = _safe_ai(table)
    table["elemental_class"] = _elemental_class(table)
    table["perminova_class"] = spec.mol_class(how="perminova").table["class"].astype(str).values
    table["vk_cell"] = [assign_vk_cell(hc, oc) for hc, oc in zip(table["H/C"], table["O/C"])]
    total_intensity = float(table["intensity"].sum())
    if total_intensity <= 0:
        raise ValueError(f"{sample_id}: non-positive retained intensity sum")

    native_count = {
        f"D{int(row.square):02d}": float(row.value)
        for row in spec.get_squares_vk(how_average="count").itertuples()
    }
    native_intensity = {
        f"D{int(row.square):02d}": float(row.value)
        for row in spec.get_squares_vk(how_average="weight").itertuples()
    }
    hum_count = hum_ms.get_squares_vk(
        table, how_average="count", square_indices="standard"
    ).set_index("square_index")["value"]
    hum_intensity = hum_ms.get_squares_vk(
        table, how_average="weight", square_indices="standard"
    ).set_index("square_index")["value"]

    features: dict[str, Any] = {
        "sample_id": sample_id,
        "class_label": class_label,
        "class_binary": int(class_label == "ADOM"),
        "ms_global__n_unique_formulas": int(len(table)),
        "ms_global__mean_molecular_mass": float(table["calc_mass"].mean()),
        "ms_global__iw_mean_molecular_mass": _weighted_mean(table["calc_mass"], table["intensity"]),
        "ms_global__mean_HC": float(table["H/C"].mean()),
        "ms_global__iw_mean_HC": _weighted_mean(table["H/C"], table["intensity"]),
        "ms_global__mean_OC": float(table["O/C"].mean()),
        "ms_global__iw_mean_OC": _weighted_mean(table["O/C"], table["intensity"]),
        "ms_global__mean_DBE": float(table["DBE"].mean()),
        "ms_global__iw_mean_DBE": _weighted_mean(table["DBE"], table["intensity"]),
        "ms_global__mean_AI": float(table["AI"].mean()),
        "ms_global__iw_mean_AI": _weighted_mean(table["AI"], table["intensity"]),
        "ms_global__AI_valid_fraction": float(table["AI"].notna().mean()),
        "ms_global__mean_NOSC": float(table["NOSC"].mean()),
        "ms_global__iw_mean_NOSC": _weighted_mean(table["NOSC"], table["intensity"]),
        "ms_global__mean_DBE_minus_O": float(table["DBE-O"].mean()),
        "ms_global__iw_mean_DBE_minus_O": _weighted_mean(table["DBE-O"], table["intensity"]),
        "ms_global__CRAM_count_fraction": float(table["CRAM"].astype(bool).mean()),
        "ms_global__CRAM_intensity_fraction": float(
            table.loc[table["CRAM"].astype(bool), "intensity"].sum() / total_intensity
        ),
    }

    elemental_rows = []
    for elemental_class in (*MAIN_ELEMENTAL_CLASSES, "OTHER_NON_OXYGENATED"):
        selected = table["elemental_class"].eq(elemental_class)
        count = int(selected.sum())
        intensity = float(table.loc[selected, "intensity"].sum())
        count_fraction = float(count / len(table))
        intensity_fraction = float(intensity / total_intensity)
        if elemental_class in MAIN_ELEMENTAL_CLASSES:
            features[f"elem__{elemental_class}__count_fraction"] = count_fraction
            features[f"elem__{elemental_class}__intensity_fraction"] = intensity_fraction
        elemental_rows.append(
            {
                "sample_id": sample_id,
                "class_label": class_label,
                "elemental_class": elemental_class,
                "count": count,
                "count_fraction_all_retained_formulas": count_fraction,
                "summed_intensity": intensity,
                "intensity_fraction_all_retained_intensity": intensity_fraction,
            }
        )

    main_elemental = table["elemental_class"].isin(MAIN_ELEMENTAL_CLASSES)
    main_count_denominator = int(main_elemental.sum())
    main_intensity_denominator = float(table.loc[main_elemental, "intensity"].sum())
    elemental_renormalized_rows = []
    for elemental_class in MAIN_ELEMENTAL_CLASSES:
        selected = table["elemental_class"].eq(elemental_class)
        elemental_renormalized_rows.append(
            {
                "sample_id": sample_id,
                "class_label": class_label,
                "elemental_class": elemental_class,
                "count_fraction_renormalized_four_classes": float(
                    selected.sum() / main_count_denominator if main_count_denominator else 0.0
                ),
                "intensity_fraction_renormalized_four_classes": float(
                    table.loc[selected, "intensity"].sum() / main_intensity_denominator
                    if main_intensity_denominator else 0.0
                ),
            }
        )

    chemotype_rows = []
    for library_name in PERMINOVA_CLASSES:
        canonical = CHEMOTYPE_CANONICAL[library_name]
        selected = table["perminova_class"].eq(library_name)
        count = int(selected.sum())
        intensity = float(table.loc[selected, "intensity"].sum())
        count_fraction = float(count / len(table))
        intensity_fraction = float(intensity / total_intensity)
        features[f"chem__{canonical}__count_fraction"] = count_fraction
        features[f"chem__{canonical}__intensity_fraction"] = intensity_fraction
        chemotype_rows.append(
            {
                "sample_id": sample_id,
                "class_label": class_label,
                "chemotype": canonical,
                "nomspectra_class": library_name,
                "count": count,
                "count_fraction": count_fraction,
                "summed_intensity": intensity,
                "intensity_fraction": intensity_fraction,
            }
        )
    undefined = ~table["perminova_class"].isin(PERMINOVA_CLASSES)

    vk_rows = []
    for number in range(1, 21):
        cell = f"D{number:02d}"
        selected = table["vk_cell"].eq(cell)
        count = int(selected.sum())
        intensity = float(table.loc[selected, "intensity"].sum())
        count_fraction = float(count / len(table))
        intensity_fraction = float(intensity / total_intensity)
        if not math.isclose(count_fraction, native_count[cell], rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"{sample_id} {cell}: NOMspectra count-grid mismatch")
        if not math.isclose(intensity_fraction, native_intensity[cell], rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"{sample_id} {cell}: NOMspectra intensity-grid mismatch")
        features[f"vk__{cell}__count_fraction"] = count_fraction
        features[f"vk__{cell}__intensity_fraction"] = intensity_fraction
        vk_rows.append(
            {
                "sample_id": sample_id,
                "class_label": class_label,
                "stable_cell_id": cell,
                "count": count,
                "count_fraction": count_fraction,
                "summed_intensity": intensity,
                "intensity_fraction": intensity_fraction,
            }
        )

    start = int(config["mass_bin_start"])
    end = int(config["mass_bin_end"])
    width = int(config["mass_bin_width"])
    edges = np.arange(start, end + width, width, dtype=float)
    masses = table["calc_mass"].to_numpy(float)
    intensities = table["intensity"].to_numpy(float)
    counts, _ = np.histogram(masses, bins=edges)
    weighted, _ = np.histogram(masses, bins=edges, weights=intensities)
    bin_rows = []
    for index, lower in enumerate(edges[:-1]):
        upper = edges[index + 1]
        label = f"{int(lower)}_{int(upper)}"
        count_fraction = float(counts[index] / len(table))
        intensity_fraction = float(weighted[index] / total_intensity)
        features[f"massbin__{label}__count"] = int(counts[index])
        features[f"massbin__{label}__intensity_fraction"] = intensity_fraction
        bin_rows.append(
            {
                "sample_id": sample_id,
                "class_label": class_label,
                "mass_bin_lower": float(lower),
                "mass_bin_upper": float(upper),
                "count": int(counts[index]),
                "count_fraction": count_fraction,
                "summed_intensity": float(weighted[index]),
                "intensity_fraction": intensity_fraction,
            }
        )

    outside = table["vk_cell"].isna()
    other_row = next(row for row in elemental_rows if row["elemental_class"] == "OTHER_NON_OXYGENATED")
    qc = {
        "sample_id": sample_id,
        "class_label": class_label,
        "n_retained_unique_formulas": int(len(table)),
        "other_non_oxygenated_count": int(other_row["count"]),
        "other_non_oxygenated_count_fraction": float(other_row["count_fraction_all_retained_formulas"]),
        "other_non_oxygenated_intensity": float(other_row["summed_intensity"]),
        "other_non_oxygenated_intensity_fraction": float(
            other_row["intensity_fraction_all_retained_intensity"]
        ),
        "four_class_count_fraction_sum_main_denominator": float(
            sum(features[f"elem__{name}__count_fraction"] for name in MAIN_ELEMENTAL_CLASSES)
        ),
        "four_class_intensity_fraction_sum_main_denominator": float(
            sum(features[f"elem__{name}__intensity_fraction"] for name in MAIN_ELEMENTAL_CLASSES)
        ),
        "four_class_count_fraction_sum_renormalized": float(
            sum(row["count_fraction_renormalized_four_classes"] for row in elemental_renormalized_rows)
        ),
        "four_class_intensity_fraction_sum_renormalized": float(
            sum(row["intensity_fraction_renormalized_four_classes"] for row in elemental_renormalized_rows)
        ),
        "outside_native_vk_grid_count": int(outside.sum()),
        "outside_native_vk_grid_count_fraction": float(outside.mean()),
        "outside_native_vk_grid_intensity_fraction": float(
            table.loc[outside, "intensity"].sum() / total_intensity
        ),
        "native_vk_count_fraction_sum": float(sum(row["count_fraction"] for row in vk_rows)),
        "native_vk_intensity_fraction_sum": float(sum(row["intensity_fraction"] for row in vk_rows)),
        "nomspectra_humspectra_count_max_abs_difference": float(
            max(abs(native_count[f"D{i:02d}"] - float(hum_count.loc[i])) for i in range(1, 21))
        ),
        "nomspectra_humspectra_intensity_max_abs_difference": float(
            max(abs(native_intensity[f"D{i:02d}"] - float(hum_intensity.loc[i])) for i in range(1, 21))
        ),
        "undefined_chemotype_count": int(undefined.sum()),
        "undefined_chemotype_count_fraction": float(undefined.mean()),
        "undefined_chemotype_intensity_fraction": float(
            table.loc[undefined, "intensity"].sum() / total_intensity
        ),
        "negative_intensity_count": int(table["intensity"].lt(0).sum()),
        "nonfinite_descriptor_count": int(
            table[["H/C", "O/C", "DBE", "DBE-O", "NOSC"]].isna().sum().sum()
        ),
    }
    return {
        "features": features,
        "elemental": elemental_rows,
        "elemental_renormalized": elemental_renormalized_rows,
        "chemotypes": chemotype_rows,
        "vk": vk_rows,
        "mass_bins": bin_rows,
        "qc": qc,
    }


def process_all_mass_files(
    raw_ms_dir: Path,
    processed_dir: Path,
    registry: pd.DataFrame,
    config: dict[str, Any],
    log,
) -> dict[str, pd.DataFrame]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    registry_index = registry.set_index("sample_id")
    collected: dict[str, list[Any]] = {
        "features": [],
        "duplicate_audit": [],
        "duplicate_summary": [],
        "elemental": [],
        "elemental_renormalized": [],
        "chemotypes": [],
        "vk": [],
        "mass_bins": [],
        "elemental_qc": [],
        "checksums": [],
    }
    paths = sorted(raw_ms_dir.glob("*.txt"))
    for number, path in enumerate(paths, start=1):
        sample_id = path.stem.rsplit("_", 1)[0]
        if sample_id not in registry_index.index:
            raise ValueError(f"{path.name}: sample is absent from the audited registry")
        class_label = str(registry_index.loc[sample_id, "class_label"])
        checksum_before = sha256_file(path)
        source = pd.read_csv(path, sep="\t")
        retained, audit, summary = resolve_duplicate_formulas(source, sample_id, config)
        result = calculate_descriptors(retained, sample_id, class_label, config)
        output = processed_dir / path.name
        retained.to_csv(output, sep="\t", index=False, encoding="utf-8", float_format="%.15g")
        checksum_after = sha256_file(path)
        if checksum_after != checksum_before:
            raise AssertionError(f"Raw MS file changed during processing: {path}")
        collected["checksums"].append(
            {
                "sample_id": sample_id,
                "raw_path": path.as_posix(),
                "raw_sha256_before": checksum_before,
                "raw_sha256_after": checksum_after,
                "raw_unchanged": checksum_before == checksum_after,
                "processed_path": output.as_posix(),
                "processed_sha256": sha256_file(output),
            }
        )
        collected["duplicate_summary"].append(summary)
        if not audit.empty:
            collected["duplicate_audit"].extend(audit.to_dict("records"))
        collected["features"].append(result["features"])
        for key in ["elemental", "elemental_renormalized", "chemotypes", "vk", "mass_bins"]:
            collected[key].extend(result[key])
        collected["elemental_qc"].append(result["qc"])
        if number % 5 == 0 or number == len(paths):
            log(f"MS preprocessing {number}/{len(paths)} complete")
    return {key: pd.DataFrame(value) for key, value in collected.items()}
