"""Reproduce the final grouped logistic-regression models.

The script uses only the public model-input tables and frozen outer-CV splits.
All preprocessing is fitted within each training fold. Scheme A uses grouped
inner CV to select C; Scheme B uses the prespecified C=1 sensitivity setting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


SEED = 429
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MODEL_NAMES = [
    "DOC_ONLY",
    "OPTICAL_RAW",
    "OPTICAL_NO_ABSOLUTE_ABS",
    "OPTICAL_DOC_NORMALIZED",
    "OPTICAL_RESIDUALIZED",
    "OPTICAL_INTENSIVE",
    "MS_COUNT_FINAL",
    "COMBINED_INTENSIVE",
]
OPTICAL_ALL = [
    "Ag_275", "Ag_380", "Asm_280", "Asm_350", "B1", "B2",
    "E2E3", "E3E4", "FI", "HIX", "S_380_443", "Suva",
]
OPTICAL_INTENSIVE = [
    "Asm_280", "Asm_350", "E2E3", "E3E4", "FI", "HIX",
    "S_380_443", "Suva",
]
METRICS = [
    "balanced_accuracy", "roc_auc", "macro_pr_auc", "sensitivity_ADOM",
    "specificity_Baikal", "mcc", "macro_f1", "brier_score", "log_loss",
]


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    prediction = (probability >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "macro_pr_auc": float(
            (average_precision_score(y_true, probability)
             + average_precision_score(1 - y_true, 1 - probability)) / 2
        ),
        "sensitivity_ADOM": float(np.mean(prediction[y_true == 1] == 1)),
        "specificity_Baikal": float(np.mean(prediction[y_true == 0] == 0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def preprocess(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train)
    test_imputed = imputer.transform(test)
    selector = VarianceThreshold(0.0)
    train_selected = selector.fit_transform(train_imputed)
    test_selected = selector.transform(test_imputed)
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train_selected),
        scaler.transform(test_selected),
        int(selector.get_support().sum()),
    )


def residualize(
    train: np.ndarray,
    test: np.ndarray,
    doc_train: np.ndarray,
    doc_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design_train = np.column_stack([np.ones(len(doc_train)), np.log10(doc_train)])
    design_test = np.column_stack([np.ones(len(doc_test)), np.log10(doc_test)])
    train_residual = np.full_like(train, np.nan, dtype=float)
    test_residual = np.full_like(test, np.nan, dtype=float)
    for column in range(train.shape[1]):
        valid_train = np.isfinite(train[:, column]) & np.isfinite(design_train).all(axis=1)
        if valid_train.sum() < 2:
            continue
        beta = np.linalg.lstsq(
            design_train[valid_train], train[valid_train, column], rcond=None
        )[0]
        train_residual[valid_train, column] = (
            train[valid_train, column] - design_train[valid_train] @ beta
        )
        valid_test = np.isfinite(test[:, column]) & np.isfinite(design_test).all(axis=1)
        test_residual[valid_test, column] = (
            test[valid_test, column] - design_test[valid_test] @ beta
        )
    return train_residual, test_residual


def feature_blocks(
    doc_train: np.ndarray,
    doc_test: np.ndarray,
    optical_train: np.ndarray,
    optical_test: np.ndarray,
    ms_train: np.ndarray,
    ms_test: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    position = {name: OPTICAL_ALL.index(name) for name in OPTICAL_ALL}
    no_abs = [position[name] for name in OPTICAL_ALL if name not in {"Ag_275", "Ag_380"}]
    intensive = [position[name] for name in OPTICAL_INTENSIVE]
    normalized_train = optical_train.copy()
    normalized_test = optical_test.copy()
    for name in ("Ag_275", "Ag_380"):
        index = position[name]
        normalized_train[:, index] /= doc_train
        normalized_test[:, index] /= doc_test
    residual_train, residual_test = residualize(
        optical_train, optical_test, doc_train, doc_test
    )
    return {
        "DOC_ONLY": (np.log10(doc_train)[:, None], np.log10(doc_test)[:, None]),
        "OPTICAL_RAW": (optical_train, optical_test),
        "OPTICAL_NO_ABSOLUTE_ABS": (
            optical_train[:, no_abs], optical_test[:, no_abs]
        ),
        "OPTICAL_DOC_NORMALIZED": (normalized_train, normalized_test),
        "OPTICAL_RESIDUALIZED": (residual_train, residual_test),
        "OPTICAL_INTENSIVE": (
            optical_train[:, intensive], optical_test[:, intensive]
        ),
        "MS_COUNT_FINAL": (ms_train, ms_test),
        "COMBINED_INTENSIVE": (
            np.column_stack([optical_train[:, intensive], ms_train]),
            np.column_stack([optical_test[:, intensive], ms_test]),
        ),
    }


def inner_splits(
    y: np.ndarray, groups: np.ndarray, max_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    group_frame = pd.DataFrame({"group": groups, "y": y}).drop_duplicates()
    n_splits = min(max_splits, int(group_frame.groupby("y")["group"].nunique().min()))
    if n_splits < 2:
        raise ValueError("At least two groups per class are required in inner CV")
    for attempt in range(100):
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + attempt
        )
        candidate = list(cv.split(np.zeros((len(y), 1)), y, groups))
        if all(
            len(np.unique(y[train])) == 2 and len(np.unique(y[valid])) == 2
            for train, valid in candidate
        ):
            return candidate
    raise ValueError("Could not create valid grouped inner splits")


def fit_probability(
    train: np.ndarray,
    y_train: np.ndarray,
    test: np.ndarray,
    c_value: float,
    seed: int,
) -> np.ndarray:
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=seed,
    ).fit(train, y_train)
    return model.predict_proba(test)[:, 1]


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = root / "data" / "processed" / "model_inputs"
    optical = pd.read_csv(inputs / "optical_descriptors.csv").sort_values("sample_id")
    ms = pd.read_csv(inputs / "mass_spectrometry_features.csv").set_index("sample_id")
    splits = pd.read_csv(inputs / "cv_splits.csv")
    if optical["sample_id"].duplicated().any() or ms.index.duplicated().any():
        raise ValueError("Duplicate sample identifiers in model inputs")
    if set(optical["sample_id"]) != set(ms.index):
        raise ValueError("Optical and mass-spectrometry model cohorts differ")
    return optical.reset_index(drop=True), ms, splits


def fit_models(root: Path, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    optical, ms_table, split_table = load_inputs(root)
    sample_ids = optical["sample_id"].tolist()
    index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    y = optical["class_binary"].to_numpy(int)
    doc = optical["DOC"].to_numpy(float)
    optical_values = optical[OPTICAL_ALL].to_numpy(float)
    ms_columns = ms_table.columns.tolist()
    ms_values = ms_table.loc[sample_ids, ms_columns].to_numpy(float)
    oof_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    selected_splits = split_table.loc[split_table["repeat"] < repeats].copy()
    for (scheme, repeat, fold), frame in selected_splits.groupby(
        ["blocking_scheme", "repeat", "outer_fold"], sort=True
    ):
        train = np.asarray(
            [index[value] for value in frame.loc[frame.role.eq("train"), "sample_id"]],
            dtype=int,
        )
        test = np.asarray(
            [index[value] for value in frame.loc[frame.role.eq("test"), "sample_id"]],
            dtype=int,
        )
        group_column = "validation_group" if scheme == "A" else "validation_group_conservative"
        groups = optical[group_column].to_numpy(str)
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Validation-group leakage detected")
        seed = SEED + (100000 if scheme == "A" else 200000) + int(repeat) * 101 + int(fold)
        outer = feature_blocks(
            doc[train], doc[test], optical_values[train], optical_values[test],
            ms_values[train], ms_values[test],
        )
        scores = {name: np.zeros(len(C_GRID), dtype=float) for name in MODEL_NAMES}
        if scheme == "A":
            for inner_train_local, inner_valid_local in inner_splits(
                y[train], groups[train], 3, seed
            ):
                inner_train = train[inner_train_local]
                inner_valid = train[inner_valid_local]
                blocks = feature_blocks(
                    doc[inner_train], doc[inner_valid],
                    optical_values[inner_train], optical_values[inner_valid],
                    ms_values[inner_train], ms_values[inner_valid],
                )
                for name in MODEL_NAMES:
                    x_train, x_valid, _ = preprocess(*blocks[name])
                    for c_index, c_value in enumerate(C_GRID):
                        probability = fit_probability(
                            x_train, y[inner_train], x_valid, c_value, seed + c_index
                        )
                        scores[name][c_index] += balanced_accuracy_score(
                            y[inner_valid], probability >= 0.5
                        )
        for name in MODEL_NAMES:
            selected_c = 1.0 if scheme == "B" else C_GRID[int(np.argmax(scores[name]))]
            x_train, x_test, retained = preprocess(*outer[name])
            probability = fit_probability(
                x_train, y[train], x_test, selected_c, seed
            )
            fold_rows.append(
                {
                    "blocking_scheme": scheme,
                    "feature_set": name,
                    "repeat": int(repeat),
                    "outer_fold": int(fold),
                    "selected_C": selected_c,
                    "retained_feature_count": retained,
                    **binary_metrics(y[test], probability),
                }
            )
            prevalence = float(y[train].mean())
            for local, position in enumerate(test):
                oof_rows.append(
                    {
                        "blocking_scheme": scheme,
                        "feature_set": name,
                        "repeat": int(repeat),
                        "outer_fold": int(fold),
                        "sample_id": sample_ids[position],
                        "validation_group": groups[position],
                        "true_class": optical.iloc[position]["class_label"],
                        "true_class_binary": int(y[position]),
                        "predicted_probability_ADOM": float(probability[local]),
                        "prevalence_reference_probability_ADOM": prevalence,
                        "selected_C": selected_c,
                    }
                )
    return pd.DataFrame(oof_rows), pd.DataFrame(fold_rows)


def summarize(oof: pd.DataFrame, bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = oof.groupby(
        ["blocking_scheme", "feature_set", "sample_id", "validation_group",
         "true_class", "true_class_binary"],
        as_index=False,
    ).agg(
        predicted_probability_ADOM=("predicted_probability_ADOM", "mean"),
        prevalence_reference_probability_ADOM=(
            "prevalence_reference_probability_ADOM", "mean"
        ),
        n_oof_predictions=("predicted_probability_ADOM", "size"),
    )
    rows = []
    for (scheme, feature_set), frame in aggregate.groupby(
        ["blocking_scheme", "feature_set"], sort=True
    ):
        observed = binary_metrics(
            frame.true_class_binary.to_numpy(int),
            frame.predicted_probability_ADOM.to_numpy(float),
        )
        row: dict[str, object] = {
            "blocking_scheme": scheme,
            "feature_set": feature_set,
            "n_samples": len(frame),
            "n_groups": frame.validation_group.nunique(),
            **observed,
        }
        if bootstrap:
            rng = np.random.default_rng(
                SEED + sum((i + 1) * ord(c) for i, c in enumerate(scheme + feature_set))
            )
            labels = frame[["validation_group", "true_class_binary"]].drop_duplicates()
            groups_by_class = {
                value: labels.loc[
                    labels.true_class_binary.eq(value), "validation_group"
                ].to_numpy()
                for value in (0, 1)
            }
            positions = {
                group: np.flatnonzero(frame.validation_group.to_numpy() == group)
                for group in labels.validation_group
            }
            distributions = {metric: [] for metric in METRICS}
            y = frame.true_class_binary.to_numpy(int)
            probability = frame.predicted_probability_ADOM.to_numpy(float)
            for _ in range(bootstrap):
                sampled: list[int] = []
                for value in (0, 1):
                    for group in rng.choice(
                        groups_by_class[value], size=len(groups_by_class[value]), replace=True
                    ):
                        sampled.extend(positions[group])
                values = binary_metrics(y[sampled], probability[sampled])
                for metric in METRICS:
                    distributions[metric].append(values[metric])
            for metric in METRICS:
                row[f"{metric}_ci_low"] = float(
                    np.quantile(distributions[metric], 0.025)
                )
                row[f"{metric}_ci_high"] = float(
                    np.quantile(distributions[metric], 0.975)
                )
        rows.append(row)
    return aggregate, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = arguments.output or root / "results" / "recomputed"
    output.mkdir(parents=True, exist_ok=True)
    oof, folds = fit_models(root, arguments.repeats)
    aggregate, performance = summarize(oof, arguments.bootstrap)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    aggregate.to_csv(output / "oof_predictions_aggregated.csv", index=False)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    performance.to_csv(output / "model_performance.csv", index=False)
    print(f"Wrote model outputs to {output}")


if __name__ == "__main__":
    main()
