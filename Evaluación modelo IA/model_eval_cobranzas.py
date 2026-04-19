from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    from sklearn.model_selection import GroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBClassifier = None  # type: ignore

RANDOM_STATE = 42
EXPECTED_TARGET_NAME = "target_mora"
RAW_TARGET_NAME = "target"
VALID_TARGET_CLASSES = {"on_time", "+30", "+60", "+90"}
VALID_CONDICION_DIAS = {30, 45, 60, 90}
RATE_COLUMNS = ["tasa_cumplimiento", "tasa_contacto_cliente", "tasa_cumpl_promesas"]
SECTOR_COLUMNS = [
    "sector_retail",
    "sector_manufactura",
    "sector_servicios",
    "sector_construccion",
    "sector_agro",
    "sector_tecnologia",
    "sector_salud",
    "sector_transporte",
]
LOG_SKEWED_COLS = [
    "monto",
    "monto_promedio_hist",
    "ratio_monto",
    "mora_promedio_hist",
    "mora_ultimo_tramo",
    "num_gestiones_factura",
    "dias_hasta_vence_pos",
    "dias_mora_observable",
    "num_no_contesta_cons",
    "num_promesas_rotas",
    "promesas_total",
    "dias_transcurridos_corte",
]
BINARY_PASSTHROUGH_COLS = [
    "tiene_garantia",
    "tiene_disputa_activa",
    "tiene_promesa_activa",
    "sin_gestion_previa",
    "esta_vencida_al_corte",
    "cliente_nuevo",
] + SECTOR_COLUMNS
COUNT_RATE_NUMERIC_COLS = [
    "condicion_dias",
    "antiguedad_meses",
    "num_facturas_prev",
    "tasa_cumplimiento",
    "moras_consecutivas",
    "tasa_contacto_cliente",
    "tasa_cumpl_promesas",
    "friccion_contacto",
    "ratio_promesas_rotas",
    "intensidad_gestion",
    "dias_desde_ultima_gestion",
]
CATEGORICAL_COLS = ["ultimo_resultado_enc"]
LIKELY_ID_COLS = {"factura_id", "cliente_id", "fecha_corte"}
LIKELY_LEAKAGE_COLS = {
    "target",
    "target_mora",
    "dias_mora_final",
    "estado_final_pago",
    "mora_final",
    "target_encoded",
}
FAIRNESS_THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmarking, fairness y diagnóstico de overfitting/underfitting para cobranzas."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Ruta al CSV principal. Puede ser features_ml.csv o features_ml_prepared.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./model_eval_outputs",
        help="Directorio donde se guardarán tablas y figuras.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=EXPECTED_TARGET_NAME,
        help="Nombre de la columna objetivo. Si no existe, se intentará usar 'target'.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporción de facturas para test.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=30,
        help="Tamaño mínimo por subgrupo para fairness.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_print(title: str, value: Any) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(value)


def load_dataset(path: Path, target_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_csv(path)
    if "fecha_corte" in df.columns:
        df["fecha_corte"] = pd.to_datetime(df["fecha_corte"], errors="coerce")
    if RAW_TARGET_NAME in df.columns and target_col not in df.columns:
        df = df.rename(columns={RAW_TARGET_NAME: target_col})
    return df


def validate_dataset(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "target_col": target_col,
        "target_exists": target_col in df.columns,
        "facturas_unicas": int(df["factura_id"].nunique()) if "factura_id" in df.columns else None,
        "clientes_unicos": int(df["cliente_id"].nunique()) if "cliente_id" in df.columns else None,
    }
    if "fecha_corte" in df.columns and not df["fecha_corte"].isna().all():
        checks["fecha_min"] = str(df["fecha_corte"].min())
        checks["fecha_max"] = str(df["fecha_corte"].max())
    if {"factura_id", "num_corte"}.issubset(df.columns):
        checks["duplicados_factura_corte"] = int(df.duplicated(subset=["factura_id", "num_corte"]).sum())
    checks["duplicados_exactos"] = int(df.duplicated().sum())
    if "monto" in df.columns:
        checks["monto_no_positivo"] = int((pd.to_numeric(df["monto"], errors="coerce") <= 0).sum())
    if "antiguedad_meses" in df.columns:
        checks["antiguedad_no_positiva"] = int((pd.to_numeric(df["antiguedad_meses"], errors="coerce") <= 0).sum())
    if "condicion_dias" in df.columns:
        checks["condicion_invalida"] = int((~df["condicion_dias"].isin(VALID_CONDICION_DIAS)).sum())
    if target_col in df.columns:
        checks["target_nulos"] = int(df[target_col].isna().sum())
        observed = set(pd.Series(df[target_col]).dropna().astype(str).unique())
        checks["clases_target_observadas"] = sorted(observed)
        checks["clases_target_invalidas"] = sorted(observed - VALID_TARGET_CLASSES)
    return checks


def clean_dataset(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()

    if {"dias_desde_ultima_gestion", "ultimo_resultado_enc"}.issubset(df.columns):
        df["sin_gestion_previa"] = (
            df["dias_desde_ultima_gestion"].isna() & df["ultimo_resultado_enc"].isna()
        ).astype(int)

        df["dias_desde_ultima_gestion"] = pd.to_numeric(
            df["dias_desde_ultima_gestion"], errors="coerce"
        ).fillna(-1)

        df["ultimo_resultado_enc"] = (
            df["ultimo_resultado_enc"].astype("Int64").astype(str).replace("<NA>", "sin_gestion_previa")
        )
        df["ultimo_resultado_enc"] = df["ultimo_resultado_enc"].apply(
            lambda x: f"cod_{x}" if x != "sin_gestion_previa" else x
        )
    elif "sin_gestion_previa" not in df.columns:
        df["sin_gestion_previa"] = 0

    redundant: Dict[str, bool] = {}
    cols_to_drop: List[str] = []

    if {"num_corte", "num_gestiones_factura"}.issubset(df.columns):
        redundant["num_corte_eq_num_gestiones_factura"] = bool(
            (pd.to_numeric(df["num_corte"], errors="coerce") == pd.to_numeric(df["num_gestiones_factura"], errors="coerce")).all()
        )
        if redundant["num_corte_eq_num_gestiones_factura"]:
            cols_to_drop.append("num_corte")

    if {"dias_hasta_vence", "condicion_dias", "dias_desde_emision"}.issubset(df.columns):
        lhs = pd.to_numeric(df["dias_hasta_vence"], errors="coerce")
        rhs = pd.to_numeric(df["condicion_dias"], errors="coerce") - pd.to_numeric(df["dias_desde_emision"], errors="coerce")
        redundant["dias_hasta_vence_eq_condicion_menos_dias_desde_emision"] = bool((lhs == rhs).all())
        if redundant["dias_hasta_vence_eq_condicion_menos_dias_desde_emision"]:
            cols_to_drop.append("dias_desde_emision")

    existing_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)

    summary: Dict[str, Any] = {
        "columnas_eliminadas": existing_drop,
        "nulos_restantes": df.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False).to_dict(),
    }
    summary.update(redundant)

    return df, summary


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"condicion_dias", "dias_hasta_vence"}.issubset(df.columns):
        df["dias_transcurridos_corte"] = pd.to_numeric(df["condicion_dias"], errors="coerce") - pd.to_numeric(df["dias_hasta_vence"], errors="coerce")
        df["esta_vencida_al_corte"] = (pd.to_numeric(df["dias_hasta_vence"], errors="coerce") < 0).astype(int)
        df["dias_mora_observable"] = np.maximum(0, -pd.to_numeric(df["dias_hasta_vence"], errors="coerce"))
        df["dias_hasta_vence_pos"] = np.maximum(0, pd.to_numeric(df["dias_hasta_vence"], errors="coerce"))

    if "num_facturas_prev" in df.columns:
        df["cliente_nuevo"] = (pd.to_numeric(df["num_facturas_prev"], errors="coerce").fillna(0) == 0).astype(int)

    if {"num_gestiones_factura", "dias_transcurridos_corte"}.issubset(df.columns):
        num_gest = pd.to_numeric(df["num_gestiones_factura"], errors="coerce").fillna(0)
        dias_tr = pd.to_numeric(df["dias_transcurridos_corte"], errors="coerce").fillna(0)
        df["intensidad_gestion"] = num_gest / np.maximum(dias_tr + 1, 1)

    if {"num_no_contesta_cons", "num_gestiones_factura"}.issubset(df.columns):
        no_cont = pd.to_numeric(df["num_no_contesta_cons"], errors="coerce").fillna(0)
        num_gest = pd.to_numeric(df["num_gestiones_factura"], errors="coerce").fillna(0)
        df["friccion_contacto"] = no_cont / np.maximum(num_gest, 1)

    if {"num_promesas_rotas", "promesas_total"}.issubset(df.columns):
        rotas = pd.to_numeric(df["num_promesas_rotas"], errors="coerce").fillna(0)
        total = pd.to_numeric(df["promesas_total"], errors="coerce").fillna(0)
        df["ratio_promesas_rotas"] = rotas / np.maximum(total, 1)

    if "sin_gestion_previa" not in df.columns:
        df["sin_gestion_previa"] = 0
    if "esta_vencida_al_corte" not in df.columns:
        df["esta_vencida_al_corte"] = 0
    if "cliente_nuevo" not in df.columns:
        df["cliente_nuevo"] = 0

    return df


def build_feature_list(df: pd.DataFrame, target_col: str) -> List[str]:
    excludes = set(LIKELY_ID_COLS) | {target_col}
    return [c for c in df.columns if c not in excludes and c not in LIKELY_LEAKAGE_COLS]


def build_preprocessor(feature_cols: List[str]) -> Tuple[ColumnTransformer, Dict[str, List[str]]]:
    numeric_log = [c for c in LOG_SKEWED_COLS if c in feature_cols]
    binary_passthrough = [c for c in BINARY_PASSTHROUGH_COLS if c in feature_cols]
    categorical = [c for c in CATEGORICAL_COLS if c in feature_cols]
    numeric_plain = [c for c in COUNT_RATE_NUMERIC_COLS if c in feature_cols]

    used = set(numeric_log + binary_passthrough + categorical + numeric_plain)
    remainder = [c for c in feature_cols if c not in used]

    numeric_log_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ]
    )
    numeric_plain_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    binary_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="sin_gestion_previa")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    remainder_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_log", numeric_log_pipe, numeric_log),
            ("num_plain", numeric_plain_pipe, numeric_plain),
            ("binary", binary_pipe, binary_passthrough),
            ("cat", categorical_pipe, categorical),
            ("remainder_num", remainder_pipe, remainder),
        ],
        remainder="drop",
    )

    schema = {
        "numeric_log": numeric_log,
        "numeric_plain": numeric_plain,
        "binary": binary_passthrough,
        "categorical": categorical,
        "remainder_numeric": remainder,
    }
    return preprocessor, schema


def split_by_factura(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"factura_id", target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas para el split por factura: {sorted(missing)}")

    factura_target = df.groupby("factura_id")[target_col].first().reset_index()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(factura_target[["factura_id"]], factura_target[target_col]))

    train_ids = factura_target.iloc[train_idx].copy()
    test_ids = factura_target.iloc[test_idx].copy()

    train_facturas = set(train_ids["factura_id"])
    test_facturas = set(test_ids["factura_id"])
    if not train_facturas.isdisjoint(test_facturas):
        raise RuntimeError("Hay facturas compartidas entre train y test. Revise el split.")

    df_train = df[df["factura_id"].isin(train_facturas)].copy()
    df_test = df[df["factura_id"].isin(test_facturas)].copy()
    return df_train, df_test, train_ids, test_ids


def compute_class_weights(y_train: pd.Series) -> Dict[str, float]:
    classes = np.array(sorted(pd.Series(y_train).astype(str).unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=pd.Series(y_train).astype(str))
    return {str(cls): float(w) for cls, w in zip(classes, weights)}


def make_models(class_weights: Dict[str, float]) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            class_weight=class_weights,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=class_weights,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return models


def make_pipeline(preprocessor: ColumnTransformer, model: Any) -> Pipeline:
    return Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", model)])


def safe_predict_proba(model: Pipeline, X: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None


def safe_roc_auc(y_true: pd.Series, y_proba: Optional[np.ndarray], classes: List[str]) -> float:
    if y_proba is None:
        return float("nan")
    y_true_bin = label_binarize(pd.Series(y_true).astype(str), classes=classes)
    if y_true_bin.shape[1] <= 1:
        return float("nan")
    try:
        return float(roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    classes: List[str],
    prefix: str,
) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    metrics = {
        f"accuracy_{prefix}": float(accuracy_score(y_true, y_pred)),
        f"balanced_accuracy_{prefix}": float(balanced_accuracy_score(y_true, y_pred)),
        f"precision_macro_{prefix}": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        f"recall_macro_{prefix}": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f"f1_macro_{prefix}": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"roc_auc_ovr_weighted_{prefix}": safe_roc_auc(y_true, y_proba, classes),
    }
    return metrics


def fit_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], str, pd.DataFrame, Dict[str, Any]]:
    classes = sorted(pd.Series(y_train).astype(str).unique())
    class_weights = compute_class_weights(y_train)
    models = make_models(class_weights)
    rows: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Pipeline] = {}
    reports: Dict[str, Any] = {}

    for model_name, estimator in models.items():
        pipe = make_pipeline(preprocessor, estimator)
        pipe.fit(X_train, y_train)
        fitted_models[model_name] = pipe

        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        y_proba_train = safe_predict_proba(pipe, X_train)
        y_proba_test = safe_predict_proba(pipe, X_test)

        row = {"model": model_name}
        row.update(evaluate_predictions(y_train, y_pred_train, y_proba_train, classes, prefix="train"))
        row.update(evaluate_predictions(y_test, y_pred_test, y_proba_test, classes, prefix="test"))
        row["gap_accuracy"] = row["accuracy_train"] - row["accuracy_test"]
        row["gap_f1_macro"] = row["f1_macro_train"] - row["f1_macro_test"]
        rows.append(row)

        reports[model_name] = {
            "classification_report_test": classification_report(y_test, y_pred_test, output_dict=True, zero_division=0),
            "classification_report_train": classification_report(y_train, y_pred_train, output_dict=True, zero_division=0),
            "confusion_matrix_test": confusion_matrix(y_test, y_pred_test, labels=classes),
            "classes": classes,
        }

    metrics_df = pd.DataFrame(rows).sort_values("f1_macro_test", ascending=False).reset_index(drop=True)

    advanced_candidates = [m for m in metrics_df["model"] if m in {"logistic_regression", "random_forest", "xgboost"}]
    best_model_name = advanced_candidates[0] if advanced_candidates else metrics_df.iloc[0]["model"]

    best_conf = pd.DataFrame(
        reports[best_model_name]["confusion_matrix_test"],
        index=reports[best_model_name]["classes"],
        columns=reports[best_model_name]["classes"],
    )
    best_conf.to_csv(output_dir / "confusion_matrix_best_model.csv", index=True)

    with open(output_dir / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

    metrics_df.to_csv(output_dir / "benchmark_metrics.csv", index=False)
    return metrics_df, fitted_models, best_model_name, best_conf, reports


def plot_benchmark_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    for metric in ["f1_macro_test", "balanced_accuracy_test"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metrics_df["model"], metrics_df[metric])
        ax.set_title(f"Comparación de modelos por {metric}")
        ax.set_xlabel("Modelo")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_dir / f"benchmark_{metric}.png", dpi=200)
        plt.close(fig)


def infer_sector_label(df: pd.DataFrame) -> pd.Series:
    available_sector_cols = [c for c in SECTOR_COLUMNS if c in df.columns]
    if not available_sector_cols:
        return pd.Series(["sin_sector"] * len(df), index=df.index)

    def row_to_sector(row: pd.Series) -> str:
        active = [col.replace("sector_", "") for col in available_sector_cols if row.get(col, 0) == 1]
        if len(active) == 1:
            return active[0]
        if len(active) > 1:
            return "multiple"
        return "sin_sector"

    return df[available_sector_cols].apply(row_to_sector, axis=1)


def false_positive_negative_rates_ovr(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    classes: List[str],
) -> Tuple[float, float]:
    y_true_s = pd.Series(list(y_true)).astype(str)
    y_pred_s = pd.Series(list(y_pred)).astype(str)
    fprs: List[float] = []
    fnrs: List[float] = []
    for cls in classes:
        true_pos = ((y_true_s == cls) & (y_pred_s == cls)).sum()
        true_neg = ((y_true_s != cls) & (y_pred_s != cls)).sum()
        false_pos = ((y_true_s != cls) & (y_pred_s == cls)).sum()
        false_neg = ((y_true_s == cls) & (y_pred_s != cls)).sum()
        fpr_den = false_pos + true_neg
        fnr_den = false_neg + true_pos
        fprs.append(float(false_pos / fpr_den) if fpr_den > 0 else float("nan"))
        fnrs.append(float(false_neg / fnr_den) if fnr_den > 0 else float("nan"))
    return float(np.nanmean(fprs)), float(np.nanmean(fnrs))


def build_fairness_views(df_test: pd.DataFrame) -> Dict[str, pd.Series]:
    views: Dict[str, pd.Series] = {}
    if "tiene_garantia" in df_test.columns:
        views["tiene_garantia"] = df_test["tiene_garantia"].map({1: "con_garantia", 0: "sin_garantia"}).fillna("desconocido")
    if "cliente_nuevo" in df_test.columns:
        views["cliente_nuevo"] = df_test["cliente_nuevo"].map({1: "nuevo", 0: "recurrente"}).fillna("desconocido")
    if "tiene_disputa_activa" in df_test.columns:
        views["tiene_disputa_activa"] = df_test["tiene_disputa_activa"].map({1: "con_disputa", 0: "sin_disputa"}).fillna("desconocido")
    if "monto" in df_test.columns:
        monto_num = pd.to_numeric(df_test["monto"], errors="coerce")
        try:
            views["bandas_monto"] = pd.qcut(monto_num, q=4, duplicates="drop").astype(str)
        except Exception:
            views["bandas_monto"] = pd.Series(["monto_no_disponible"] * len(df_test), index=df_test.index)
    if "intensidad_gestion" in df_test.columns:
        intensidad_num = pd.to_numeric(df_test["intensidad_gestion"], errors="coerce")
        try:
            views["bandas_intensidad_gestion"] = pd.qcut(intensidad_num, q=4, duplicates="drop").astype(str)
        except Exception:
            views["bandas_intensidad_gestion"] = pd.Series(["intensidad_no_disponible"] * len(df_test), index=df_test.index)
    sector = infer_sector_label(df_test)
    if sector.nunique() > 1:
        views["sector"] = sector
    return views


def run_fairness_analysis(
    best_model: Pipeline,
    best_model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test_raw: pd.DataFrame,
    output_dir: Path,
    min_group_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    classes = sorted(pd.Series(y_test).astype(str).unique())
    y_pred = pd.Series(best_model.predict(X_test), index=X_test.index).astype(str)
    fairness_views = build_fairness_views(df_test_raw)
    fairness_rows: List[Dict[str, Any]] = []

    for view_name, group_series in fairness_views.items():
        aligned_groups = pd.Series(group_series, index=X_test.index).astype(str)
        for group_value, idx in aligned_groups.groupby(aligned_groups).groups.items():
            subgroup_idx = list(idx)
            if len(subgroup_idx) < min_group_size:
                continue
            y_true_g = pd.Series(y_test.loc[subgroup_idx]).astype(str)
            y_pred_g = y_pred.loc[subgroup_idx]
            fpr, fnr = false_positive_negative_rates_ovr(y_true_g, y_pred_g, classes)
            pred_dist = y_pred_g.value_counts(normalize=True).round(4).to_dict()
            fairness_rows.append(
                {
                    "model": best_model_name,
                    "group_variable": view_name,
                    "group_value": group_value,
                    "group_size": len(subgroup_idx),
                    "accuracy": float(accuracy_score(y_true_g, y_pred_g)),
                    "precision_macro": float(precision_score(y_true_g, y_pred_g, average="macro", zero_division=0)),
                    "recall_macro": float(recall_score(y_true_g, y_pred_g, average="macro", zero_division=0)),
                    "f1_macro": float(f1_score(y_true_g, y_pred_g, average="macro", zero_division=0)),
                    "false_positive_rate_macro": fpr,
                    "false_negative_rate_macro": fnr,
                    "prediction_distribution": json.dumps(pred_dist, ensure_ascii=False),
                }
            )

    fairness_df = pd.DataFrame(fairness_rows)
    fairness_df.to_csv(output_dir / "fairness_by_group.csv", index=False)

    gap_rows: List[Dict[str, Any]] = []
    if not fairness_df.empty:
        for group_var, subdf in fairness_df.groupby("group_variable"):
            for metric in ["f1_macro", "recall_macro", "false_positive_rate_macro", "false_negative_rate_macro"]:
                max_val = subdf[metric].max()
                min_val = subdf[metric].min()
                gap = max_val - min_val
                gap_rows.append(
                    {
                        "group_variable": group_var,
                        "metric": metric,
                        "max_value": float(max_val),
                        "min_value": float(min_val),
                        "max_gap": float(gap),
                        "warning_gap_gt_threshold": bool(gap > FAIRNESS_THRESHOLD),
                    }
                )
            for metric in ["f1_macro", "recall_macro"]:
                plot_df = subdf.sort_values(metric, ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(plot_df["group_value"], plot_df[metric])
                ax.set_title(f"{metric} por grupo - {group_var}")
                ax.set_xlabel("Grupo")
                ax.set_ylabel(metric)
                ax.tick_params(axis="x", rotation=30)
                fig.tight_layout()
                fig.savefig(output_dir / f"fairness_{group_var}_{metric}.png", dpi=200)
                plt.close(fig)

    fairness_gap_df = pd.DataFrame(gap_rows)
    fairness_gap_df.to_csv(output_dir / "fairness_gap_summary.csv", index=False)
    return fairness_df, fairness_gap_df


def make_group_cv(n_splits: int = 5):
    if HAS_STRATIFIED_GROUP_KFOLD:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return GroupKFold(n_splits=n_splits)


def plot_learning_curve_single(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_name: str,
    output_dir: Path,
) -> Dict[str, float]:
    cv = make_group_cv(n_splits=5)
    train_sizes = np.linspace(0.2, 1.0, 5)
    sizes, train_scores, valid_scores = learning_curve(
        estimator,
        X,
        y,
        groups=groups,
        train_sizes=train_sizes,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sizes, train_mean, marker="o", label="Train F1-macro")
    ax.plot(sizes, valid_mean, marker="o", label="Validation F1-macro")
    ax.set_title(f"Curva de aprendizaje - {model_name}")
    ax.set_xlabel("Tamaño de entrenamiento")
    ax.set_ylabel("F1-macro")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"learning_curve_{model_name}.png", dpi=200)
    plt.close(fig)

    final_gap = float(train_mean[-1] - valid_mean[-1])
    if train_mean[-1] > 0.80 and final_gap > 0.10:
        diagnosis = "posible_overfitting"
    elif train_mean[-1] < 0.60 and valid_mean[-1] < 0.60:
        diagnosis = "posible_underfitting"
    else:
        diagnosis = "ajuste_aceptable_o_mixto"

    return {
        "model": model_name,
        "learning_curve_train_final": float(train_mean[-1]),
        "learning_curve_validation_final": float(valid_mean[-1]),
        "learning_curve_gap_final": final_gap,
        "learning_curve_diagnosis": diagnosis,
    }


def run_overfitting_analysis(
    fitted_models: Dict[str, Pipeline],
    metrics_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    output_dir: Path,
) -> pd.DataFrame:
    target_models = [m for m in ["logistic_regression", "random_forest", "xgboost"] if m in fitted_models]
    gap_base = metrics_df[[
        "model",
        "accuracy_train",
        "accuracy_test",
        "f1_macro_train",
        "f1_macro_test",
        "gap_accuracy",
        "gap_f1_macro",
    ]].copy()
    curve_rows: List[Dict[str, Any]] = []
    for model_name in target_models:
        row = plot_learning_curve_single(
            fitted_models[model_name], X_train, y_train, train_groups, model_name, output_dir
        )
        curve_rows.append(row)

    curve_df = pd.DataFrame(curve_rows)
    gap_df = gap_base.merge(curve_df, on="model", how="left")
    gap_df.to_csv(output_dir / "train_test_gap.csv", index=False)
    return gap_df


def export_state_of_art_template(output_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "autor_anio": "",
                "dominio": "",
                "tamano_o_tipo_datos": "",
                "modelo": "",
                "metrica_principal": "",
                "resultado_reportado": "",
                "comparabilidad": "",
                "observaciones": "",
            }
            for _ in range(3)
        ]
    )
    df.to_csv(output_dir / "sota_comparison_template.csv", index=False)
    return df


def export_commercial_template(output_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "opcion": "modelo_propio_xgboost",
                "tipo": "propio",
                "personalizacion": "alta",
                "explicabilidad": "media",
                "costo_estimado": "infra propia / bajo a medio",
                "dependencia_proveedor": "baja",
                "privacidad": "alta si se mantiene on-premise o en entorno controlado",
                "facilidad_despliegue": "media",
                "observaciones": "alineable con reglas de negocio y datos internos",
            },
            {
                "opcion": "google_vertex_ai",
                "tipo": "comercial",
                "personalizacion": "media-alta",
                "explicabilidad": "media",
                "costo_estimado": "variable por uso",
                "dependencia_proveedor": "alta",
                "privacidad": "depende de arquitectura y gobierno del dato",
                "facilidad_despliegue": "alta",
                "observaciones": "útil para despliegue rápido y MLOps gestionado",
            },
            {
                "opcion": "azure_machine_learning",
                "tipo": "comercial",
                "personalizacion": "media-alta",
                "explicabilidad": "media",
                "costo_estimado": "variable por uso",
                "dependencia_proveedor": "alta",
                "privacidad": "depende de configuración empresarial",
                "facilidad_despliegue": "alta",
                "observaciones": "integración fuerte con ecosistema Microsoft",
            },
            {
                "opcion": "aws_sagemaker",
                "tipo": "comercial",
                "personalizacion": "alta",
                "explicabilidad": "media",
                "costo_estimado": "variable por uso",
                "dependencia_proveedor": "alta",
                "privacidad": "depende del control de nube y gobierno de acceso",
                "facilidad_despliegue": "alta",
                "observaciones": "robusto para producción pero exige gobierno técnico",
            },
        ]
    )
    df.to_csv(output_dir / "commercial_comparison_template.csv", index=False)
    return df


def build_summary_text(
    metrics_df: pd.DataFrame,
    best_model_name: str,
    fairness_gap_df: pd.DataFrame,
    gap_df: pd.DataFrame,
) -> str:
    baseline_candidates = metrics_df[metrics_df["model"].isin(["dummy_most_frequent", "dummy_stratified", "logistic_regression"])]
    best_baseline = baseline_candidates.sort_values("f1_macro_test", ascending=False).iloc[0]["model"] if not baseline_candidates.empty else "no_disponible"
    best_advanced = best_model_name

    summary_lines = []
    summary_lines.append(f"Mejor baseline por F1-macro en test: {best_baseline}.")
    summary_lines.append(f"Mejor modelo avanzado seleccionado: {best_advanced}.")

    if not gap_df.empty:
        overfit_signals = gap_df.sort_values("gap_f1_macro", ascending=False)
        strongest = overfit_signals.iloc[0]
        summary_lines.append(
            f"Mayor brecha train-test en F1-macro: {strongest['model']} con {strongest['gap_f1_macro']:.4f}."
        )
        best_row = gap_df[gap_df["model"] == best_model_name]
        if not best_row.empty:
            diagnosis = best_row.iloc[0].get("learning_curve_diagnosis", "sin_diagnostico")
            summary_lines.append(f"Diagnóstico de curva para {best_model_name}: {diagnosis}.")

    if not fairness_gap_df.empty:
        warnings = fairness_gap_df[fairness_gap_df["warning_gap_gt_threshold"] == True]
        if not warnings.empty:
            top_warning = warnings.sort_values("max_gap", ascending=False).iloc[0]
            summary_lines.append(
                f"Mayor brecha de fairness detectada en {top_warning['group_variable']} para {top_warning['metric']} con {top_warning['max_gap']:.4f}."
            )
        else:
            summary_lines.append("No se detectaron brechas de fairness por encima del umbral configurado.")
    else:
        summary_lines.append("No fue posible calcular fairness por grupos con el tamaño mínimo configurado.")

    summary_lines.append(
        "Recomendaciones iniciales: revisar variables proxy, monitorear métricas por grupo, mantener split por factura_id y priorizar F1-macro junto con balanced accuracy y recall macro."
    )
    return "\n".join(summary_lines)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df_raw = load_dataset(data_path, args.target_col)
    validations = validate_dataset(df_raw, args.target_col)
    safe_print("Validaciones iniciales", pd.Series(validations))

    target_col = args.target_col
    if target_col not in df_raw.columns:
        raise ValueError(f"No se encontró la columna target '{target_col}' en el dataset.")
    if "factura_id" not in df_raw.columns:
        raise ValueError("No se encontró la columna 'factura_id', necesaria para evitar leakage.")

    df_clean, cleaning_summary = clean_dataset(df_raw, target_col)
    df_prepared = engineer_features(df_clean)
    feature_cols = build_feature_list(df_prepared, target_col)

    df_train, df_test, train_ids, test_ids = split_by_factura(
        df_prepared,
        target_col=target_col,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )

    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    y_train = df_train[target_col].astype(str).copy()
    y_test = df_test[target_col].astype(str).copy()

    overlap = set(train_ids["factura_id"]).intersection(set(test_ids["factura_id"]))
    if overlap:
        raise RuntimeError("Se detectó intersección de facturas entre train y test.")

    split_summary = pd.DataFrame(
        [
            {
                "filas_train": len(df_train),
                "filas_test": len(df_test),
                "facturas_train": train_ids["factura_id"].nunique(),
                "facturas_test": test_ids["factura_id"].nunique(),
                "interseccion_facturas": len(overlap),
            }
        ]
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)

    safe_print("Resumen de split", split_summary)
    safe_print("Distribución target train", y_train.value_counts(normalize=True).round(4))
    safe_print("Distribución target test", y_test.value_counts(normalize=True).round(4))

    preprocessor, schema = build_preprocessor(feature_cols)
    with open(output_dir / "preprocessing_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    metrics_df, fitted_models, best_model_name, best_conf, reports = fit_and_evaluate_models(
        X_train, y_train, X_test, y_test, preprocessor, output_dir
    )
    plot_benchmark_metrics(metrics_df, output_dir)

    best_model = fitted_models[best_model_name]
    fairness_df, fairness_gap_df = run_fairness_analysis(
        best_model=best_model,
        best_model_name=best_model_name,
        X_test=X_test,
        y_test=y_test,
        df_test_raw=df_test,
        output_dir=output_dir,
        min_group_size=args.min_group_size,
    )

    train_groups = df_train["factura_id"].astype(str)
    gap_df = run_overfitting_analysis(
        fitted_models=fitted_models,
        metrics_df=metrics_df,
        X_train=X_train,
        y_train=y_train,
        train_groups=train_groups,
        output_dir=output_dir,
    )

    export_state_of_art_template(output_dir)
    export_commercial_template(output_dir)

    artifacts_summary = {
        "input_path": str(data_path),
        "output_dir": str(output_dir),
        "best_model_name": best_model_name,
        "feature_count": len(feature_cols),
        "cleaning_summary": cleaning_summary,
        "class_weights": compute_class_weights(y_train),
        "xgboost_available": HAS_XGBOOST,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(artifacts_summary, f, indent=2, ensure_ascii=False)

    summary_text = build_summary_text(metrics_df, best_model_name, fairness_gap_df, gap_df)
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    safe_print("Top benchmarking", metrics_df[["model", "f1_macro_test", "balanced_accuracy_test", "gap_f1_macro"]])
    if not fairness_gap_df.empty:
        safe_print("Brechas fairness", fairness_gap_df.sort_values("max_gap", ascending=False).head(10))
    safe_print("Diagnóstico train-test", gap_df)
    safe_print("Resumen automático", summary_text)
    print(f"\nArtefactos guardados en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
