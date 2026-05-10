"""Exploracion visual y didactica de predicciones por factura.

Este script no reentrena modelos. Usa el artefacto exportado por
`evaluacion_modelos_cobranzas.py` para explicar, con datos concretos, como se
comporta el modelo seleccionado sobre test y sobre escenarios nuevos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PREP_OUTPUTS = PROJECT_DIR / "03_preparacion" / "outputs"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PREP_OUTPUTS / "features_ml_prepared.csv"
TEST_IDS_PATH = PREP_OUTPUTS / "test_facturas_ids.csv"
MODEL_PATH = OUTPUT_DIR / "best_model_artifact.joblib"
FEATURE_SCHEMA_PATH = OUTPUT_DIR / "model_feature_schema.csv"
METADATA_PATH = OUTPUT_DIR / "model_metadata.json"

TARGET_COL = "target_mora"
ID_COL = "factura_id"
CLIENT_COL = "cliente_id"
DATE_COL = "fecha_corte"
HIGH_RISK_CLASSES = {"+60", "+90", "60", "90"}
LATE_CLASS_WEIGHTS = {
    "+30": 0.40,
    "30": 0.40,
    "+60": 0.70,
    "60": 0.70,
    "+90": 1.00,
    "90": 1.00,
}
LOG1P_COLS = [
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

PROBA_ROWS_PATH = OUTPUT_DIR / "prediction_test_rows.csv"
INVOICE_SUMMARY_PATH = OUTPUT_DIR / "prediction_invoice_summary.csv"
TIMELINE_EXAMPLES_PATH = OUTPUT_DIR / "prediction_invoice_timeline_examples.csv"
NEW_SCENARIO_INPUTS_PATH = OUTPUT_DIR / "prediction_new_invoice_inputs.csv"
NEW_SCENARIOS_PATH = OUTPUT_DIR / "prediction_new_invoice_scenarios.csv"
FEATURE_DICTIONARY_PATH = OUTPUT_DIR / "prediction_feature_dictionary.csv"
MODEL_PARAMETERS_PATH = OUTPUT_DIR / "prediction_model_parameters.csv"
TOP_PARAMETERS_PATH = OUTPUT_DIR / "prediction_top_model_parameters.csv"


def require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        sys.exit(f"[ERROR] Faltan archivos requeridos:\n{missing_text}")


def label_slug(label: str) -> str:
    return (
        str(label)
        .replace("+", "plus_")
        .replace("-", "minus_")
        .replace(" ", "_")
        .replace("/", "_")
        .lower()
    )


def decode_labels(values, target_encoder) -> np.ndarray:
    values_array = np.asarray(values)
    if target_encoder is None:
        return values_array.astype(str)
    return target_encoder.inverse_transform(values_array.astype(int)).astype(str)


def class_names_from_model(model, target_encoder) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        if target_encoder is not None:
            return [str(c) for c in target_encoder.classes_]
        return []
    return decode_labels(classes, target_encoder).tolist()


def add_prediction_columns(
    source_df: pd.DataFrame,
    artifact: dict,
    class_names: list[str],
) -> pd.DataFrame:
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    feature_cols = artifact["feature_cols"]
    target_encoder = artifact.get("target_encoder")

    missing_features = [col for col in feature_cols if col not in source_df.columns]
    if missing_features:
        raise ValueError(f"Faltan features para predecir: {missing_features}")

    x_matrix = preprocessor.transform(source_df[feature_cols])
    raw_pred = model.predict(x_matrix)
    pred_labels = decode_labels(raw_pred, target_encoder)

    result = source_df.copy()
    result["predicted_class"] = pred_labels

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_matrix)
        for idx, class_name in enumerate(class_names):
            result[f"prob_{label_slug(class_name)}"] = proba[:, idx]
        proba_cols = [f"prob_{label_slug(c)}" for c in class_names]
        result["confidence_probability"] = result[proba_cols].max(axis=1)
        high_cols = [
            f"prob_{label_slug(c)}"
            for c in class_names
            if str(c).strip().lower() in HIGH_RISK_CLASSES
        ]
        late_cols = [
            f"prob_{label_slug(c)}"
            for c in class_names
            if str(c).strip().lower() in LATE_CLASS_WEIGHTS
        ]
        result["high_risk_probability"] = (
            result[high_cols].sum(axis=1) if high_cols else np.nan
        )
        result["any_late_probability"] = (
            result[late_cols].sum(axis=1) if late_cols else np.nan
        )
        weighted_priority = pd.Series(0.0, index=result.index)
        for class_name in class_names:
            weight = LATE_CLASS_WEIGHTS.get(str(class_name).strip().lower())
            if weight is not None:
                weighted_priority = weighted_priority + result[f"prob_{label_slug(class_name)}"] * weight
        result["priority_score_0_100"] = weighted_priority.mul(100).round(2)
    else:
        result["confidence_probability"] = np.nan
        result["high_risk_probability"] = np.nan
        result["any_late_probability"] = np.nan
        result["priority_score_0_100"] = np.nan

    if TARGET_COL in result.columns:
        result["actual_class"] = result[TARGET_COL].astype(str)
        result["is_correct"] = result["actual_class"].eq(result["predicted_class"])

    return result


def latest_cut_per_invoice(pred_rows: pd.DataFrame) -> pd.DataFrame:
    df = pred_rows.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    latest = (
        df.sort_values([ID_COL, DATE_COL])
        .groupby(ID_COL, as_index=False)
        .tail(1)
        .sort_values("priority_score_0_100", ascending=False)
        .reset_index(drop=True)
    )
    return latest


def choose_timeline_examples(pred_rows: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    selected: list[tuple[str, str]] = []

    def add_from(frame: pd.DataFrame, label: str, factura_col: str = ID_COL) -> None:
        for factura_id in frame[factura_col].astype(str).tolist():
            if factura_id not in [fid for fid, _ in selected]:
                selected.append((factura_id, label))
                return

    correct = latest[latest["is_correct"]].sort_values(
        "confidence_probability", ascending=False
    )
    wrong = latest[~latest["is_correct"]].sort_values(
        "confidence_probability", ascending=False
    )
    high_risk = latest.sort_values("priority_score_0_100", ascending=False)
    low_risk = latest.sort_values("priority_score_0_100", ascending=True)

    risk_range = (
        pred_rows.groupby(ID_COL)["priority_score_0_100"]
        .agg(["min", "max"])
        .assign(risk_change=lambda x: x["max"] - x["min"])
        .sort_values("risk_change", ascending=False)
        .reset_index()
    )

    add_from(correct, "acierto_alta_confianza")
    add_from(wrong, "error_alta_confianza")
    add_from(high_risk, "alto_riesgo_predicho")
    add_from(low_risk, "bajo_riesgo_predicho")
    add_from(risk_range, "riesgo_cambia_entre_cortes")

    selected_ids = [fid for fid, _ in selected]
    labels = dict(selected)
    examples = pred_rows[pred_rows[ID_COL].astype(str).isin(selected_ids)].copy()
    examples["example_type"] = examples[ID_COL].astype(str).map(labels)
    examples[DATE_COL] = pd.to_datetime(examples[DATE_COL], errors="coerce")
    examples = examples.sort_values(["example_type", ID_COL, DATE_COL]).reset_index(drop=True)
    return examples


def numeric_profile(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    profile = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            profile[col] = float(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            profile[col] = str(mode.iloc[0]) if len(mode) else ""
    return profile


def apply_sector(row: dict, sector_name: str) -> None:
    sector_cols = [col for col in row if col.startswith("sector_")]
    for col in sector_cols:
        row[col] = 1 if col == f"sector_{sector_name}" else 0


def build_new_invoice_scenarios(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    base = numeric_profile(df, feature_cols)
    median_monto = float(df["monto"].median()) if "monto" in df.columns else 10000.0
    median_hist_monto = (
        float(df["monto_promedio_hist"].median())
        if "monto_promedio_hist" in df.columns
        else median_monto
    )

    scenarios: list[dict] = []

    preventive = base.copy()
    preventive.update(
        {
            "factura_id": "NUEVA_PREVENTIVA_BUEN_HISTORIAL",
            "cliente_id": "CLI_NUEVO_EJEMPLO_01",
            "fecha_corte": "2026-05-09",
            "monto": round(median_monto, 2),
            "condicion_dias": 60,
            "antiguedad_meses": 24,
            "tiene_garantia": 1,
            "num_facturas_prev": 12,
            "mora_promedio_hist": 0.0,
            "mora_ultimo_tramo": 0.0,
            "tasa_cumplimiento": 0.95,
            "monto_promedio_hist": round(median_hist_monto, 2),
            "ratio_monto": 1.0,
            "moras_consecutivas": 0,
            "num_gestiones_factura": 0,
            "dias_desde_ultima_gestion": -1,
            "dias_hasta_vence": 30,
            "tasa_contacto_cliente": 0.90,
            "ultimo_resultado_enc": "cod_nan",
            "num_no_contesta_cons": 0,
            "tiene_disputa_activa": 0,
            "tiene_promesa_activa": 0,
            "num_promesas_rotas": 0,
            "tasa_cumpl_promesas": 1.0,
            "promesas_total": 3,
            "sin_gestion_previa": 1,
            "dias_transcurridos_corte": 30,
            "esta_vencida_al_corte": 0,
            "dias_mora_observable": 0,
            "dias_hasta_vence_pos": 30,
            "cliente_nuevo": 0,
            "intensidad_gestion": 0.0,
            "friccion_contacto": 0.0,
            "ratio_promesas_rotas": 0.0,
            "escenario": "Factura preventiva con buen historial",
            "lectura_esperada": "Deberia tender a bajo riesgo si el historial pesa mas que el monto.",
        }
    )
    apply_sector(preventive, "servicios")
    scenarios.append(preventive)

    new_client = base.copy()
    new_client.update(
        {
            "factura_id": "NUEVA_CLIENTE_SIN_HISTORIA",
            "cliente_id": "CLI_NUEVO_EJEMPLO_02",
            "fecha_corte": "2026-05-09",
            "monto": round(median_monto * 0.8, 2),
            "condicion_dias": 45,
            "antiguedad_meses": 1,
            "tiene_garantia": 0,
            "num_facturas_prev": 0,
            "mora_promedio_hist": 0.0,
            "mora_ultimo_tramo": 0.0,
            "tasa_cumplimiento": 0.5,
            "monto_promedio_hist": round(median_monto * 0.8, 2),
            "ratio_monto": 1.0,
            "moras_consecutivas": 0,
            "num_gestiones_factura": 0,
            "dias_desde_ultima_gestion": -1,
            "dias_hasta_vence": 20,
            "tasa_contacto_cliente": 0.5,
            "ultimo_resultado_enc": "cod_nan",
            "num_no_contesta_cons": 0,
            "tiene_disputa_activa": 0,
            "tiene_promesa_activa": 0,
            "num_promesas_rotas": 0,
            "tasa_cumpl_promesas": 0.5,
            "promesas_total": 0,
            "sin_gestion_previa": 1,
            "dias_transcurridos_corte": 25,
            "esta_vencida_al_corte": 0,
            "dias_mora_observable": 0,
            "dias_hasta_vence_pos": 20,
            "cliente_nuevo": 1,
            "intensidad_gestion": 0.0,
            "friccion_contacto": 0.0,
            "ratio_promesas_rotas": 0.0,
            "escenario": "Factura nueva sin historial",
            "lectura_esperada": "Deberia reflejar incertidumbre porque no hay historial real del cliente.",
        }
    )
    apply_sector(new_client, "retail")
    scenarios.append(new_client)

    difficult_contact = base.copy()
    difficult_contact.update(
        {
            "factura_id": "NUEVA_VENCIDA_CONTACTO_DIFICIL",
            "cliente_id": "CLI_NUEVO_EJEMPLO_03",
            "fecha_corte": "2026-05-09",
            "monto": round(median_monto * 1.4, 2),
            "condicion_dias": 30,
            "antiguedad_meses": 18,
            "tiene_garantia": 0,
            "num_facturas_prev": 8,
            "mora_promedio_hist": 18.0,
            "mora_ultimo_tramo": 35.0,
            "tasa_cumplimiento": 0.45,
            "monto_promedio_hist": round(median_hist_monto, 2),
            "ratio_monto": 1.4,
            "moras_consecutivas": 2,
            "num_gestiones_factura": 4,
            "dias_desde_ultima_gestion": 5,
            "dias_hasta_vence": -20,
            "tasa_contacto_cliente": 0.25,
            "ultimo_resultado_enc": "cod_4",
            "num_no_contesta_cons": 3,
            "tiene_disputa_activa": 1,
            "tiene_promesa_activa": 1,
            "num_promesas_rotas": 2,
            "tasa_cumpl_promesas": 0.25,
            "promesas_total": 4,
            "sin_gestion_previa": 0,
            "dias_transcurridos_corte": 50,
            "esta_vencida_al_corte": 1,
            "dias_mora_observable": 20,
            "dias_hasta_vence_pos": 0,
            "cliente_nuevo": 0,
            "intensidad_gestion": 4 / 51,
            "friccion_contacto": 0.75,
            "ratio_promesas_rotas": 0.5,
            "escenario": "Factura vencida con contacto dificil",
            "lectura_esperada": "Deberia subir el riesgo por mora observable, friccion y promesas rotas.",
        }
    )
    apply_sector(difficult_contact, "transporte")
    scenarios.append(difficult_contact)

    critical = base.copy()
    critical.update(
        {
            "factura_id": "NUEVA_CRITICA_MORA_SEVERA",
            "cliente_id": "CLI_NUEVO_EJEMPLO_04",
            "fecha_corte": "2026-05-09",
            "monto": round(median_monto * 2.2, 2),
            "condicion_dias": 30,
            "antiguedad_meses": 36,
            "tiene_garantia": 0,
            "num_facturas_prev": 20,
            "mora_promedio_hist": 45.0,
            "mora_ultimo_tramo": 90.0,
            "tasa_cumplimiento": 0.20,
            "monto_promedio_hist": round(median_hist_monto, 2),
            "ratio_monto": 2.2,
            "moras_consecutivas": 5,
            "num_gestiones_factura": 8,
            "dias_desde_ultima_gestion": 3,
            "dias_hasta_vence": -95,
            "tasa_contacto_cliente": 0.15,
            "ultimo_resultado_enc": "cod_4",
            "num_no_contesta_cons": 6,
            "tiene_disputa_activa": 1,
            "tiene_promesa_activa": 1,
            "num_promesas_rotas": 5,
            "tasa_cumpl_promesas": 0.10,
            "promesas_total": 6,
            "sin_gestion_previa": 0,
            "dias_transcurridos_corte": 125,
            "esta_vencida_al_corte": 1,
            "dias_mora_observable": 95,
            "dias_hasta_vence_pos": 0,
            "cliente_nuevo": 0,
            "intensidad_gestion": 8 / 126,
            "friccion_contacto": 0.85,
            "ratio_promesas_rotas": 5 / 6,
            "escenario": "Factura critica con mora severa",
            "lectura_esperada": "Deberia concentrar probabilidad en clases severas si el modelo captura riesgo operativo.",
        }
    )
    apply_sector(critical, "construccion")
    scenarios.append(critical)

    scenario_df = pd.DataFrame(scenarios)
    return scenario_df


def build_feature_dictionary(feature_cols: list[str]) -> pd.DataFrame:
    meanings = {
        "monto": "Valor monetario de la factura.",
        "condicion_dias": "Plazo pactado de pago en dias.",
        "antiguedad_meses": "Antiguedad del cliente en meses.",
        "tiene_garantia": "Indica si la factura o cliente tiene garantia.",
        "num_facturas_prev": "Cantidad de facturas previas del cliente.",
        "mora_promedio_hist": "Mora promedio historica del cliente.",
        "mora_ultimo_tramo": "Mora mas reciente observada en el historial.",
        "tasa_cumplimiento": "Proporcion historica de cumplimiento de pago.",
        "monto_promedio_hist": "Monto promedio historico del cliente.",
        "ratio_monto": "Monto actual comparado contra el monto historico del cliente.",
        "moras_consecutivas": "Cantidad de moras consecutivas previas.",
        "num_gestiones_factura": "Gestiones realizadas sobre la factura hasta el corte.",
        "dias_desde_ultima_gestion": "Dias desde la ultima gestion; -1 significa sin gestion previa.",
        "dias_hasta_vence": "Dias restantes hasta vencimiento; negativo significa vencida.",
        "tasa_contacto_cliente": "Tasa historica de contacto efectivo.",
        "ultimo_resultado_enc": "Resultado codificado de la ultima gestion.",
        "num_no_contesta_cons": "No contestaciones consecutivas.",
        "tiene_disputa_activa": "Indica disputa activa.",
        "tiene_promesa_activa": "Indica promesa de pago activa.",
        "num_promesas_rotas": "Promesas de pago incumplidas.",
        "tasa_cumpl_promesas": "Proporcion de promesas cumplidas.",
        "promesas_total": "Total historico de promesas.",
        "sin_gestion_previa": "Marca el primer corte sin gestion previa.",
        "dias_transcurridos_corte": "Dias transcurridos desde emision hasta el corte.",
        "esta_vencida_al_corte": "Indica si ya estaba vencida en el corte.",
        "dias_mora_observable": "Dias de mora observables al corte.",
        "dias_hasta_vence_pos": "Dias restantes si no ha vencido; cero si ya vencio.",
        "cliente_nuevo": "Cliente sin facturas previas.",
        "intensidad_gestion": "Gestiones relativas al tiempo transcurrido.",
        "friccion_contacto": "Dificultad de contacto observada.",
        "ratio_promesas_rotas": "Promesas rotas sobre promesas totales.",
    }
    rows = []
    for col in feature_cols:
        if col.startswith("sector_"):
            group = "sector"
            meaning = f"Marca one-hot de sector: {col.replace('sector_', '')}."
            expected_effect = "Captura diferencias historicas de comportamiento por industria."
        elif col in {"tiene_garantia", "tiene_disputa_activa", "tiene_promesa_activa", "sin_gestion_previa", "esta_vencida_al_corte", "cliente_nuevo"}:
            group = "binaria"
            meaning = meanings.get(col, "Indicador binario.")
            expected_effect = "El modelo aprende si el evento aumenta o reduce la probabilidad de cada clase."
        elif col == "ultimo_resultado_enc":
            group = "categorica"
            meaning = meanings.get(col, "Categoria codificada.")
            expected_effect = "Se transforma con one-hot; cada resultado puede empujar una clase distinta."
        else:
            group = "numerica"
            meaning = meanings.get(col, "Variable numerica usada por el modelo.")
            expected_effect = "Valores mayores o menores cambian la probabilidad segun el patron aprendido."
        rows.append(
            {
                "feature": col,
                "group": group,
                "business_meaning": meaning,
                "how_to_read_it": expected_effect,
            }
        )
    return pd.DataFrame(rows)


def transformed_feature_names(
    artifact: dict,
    source_df: pd.DataFrame,
    feature_cols: list[str],
) -> list[str]:
    preprocessor = artifact["preprocessor"]
    log1p_cols = [
        c
        for c in LOG1P_COLS
        if c in feature_cols and pd.api.types.is_numeric_dtype(source_df[c])
    ]
    nominal_cols = [c for c in ["ultimo_resultado_enc"] if c in feature_cols]
    binary_cols = [
        c
        for c in feature_cols
        if c not in log1p_cols
        and c not in nominal_cols
        and pd.api.types.is_numeric_dtype(source_df[c])
        and set(source_df[c].dropna().unique()).issubset({0, 1, True, False, 0.0, 1.0})
    ]
    numeric_cols = [
        c
        for c in feature_cols
        if c not in log1p_cols
        and c not in binary_cols
        and c not in nominal_cols
        and pd.api.types.is_numeric_dtype(source_df[c])
    ]

    names = [f"log1p_scaled__{c}" for c in log1p_cols]
    names.extend(f"numeric_scaled__{c}" for c in numeric_cols)
    names.extend(f"binary__{c}" for c in binary_cols)

    if nominal_cols:
        try:
            ohe = preprocessor.named_transformers_["nominal"].named_steps["ohe"]
            for col, categories in zip(nominal_cols, ohe.categories_):
                names.extend(f"onehot__{col}_{cat}" for cat in categories)
        except Exception:
            names.extend(f"onehot__{col}" for col in nominal_cols)

    return names


def export_model_parameters(
    artifact: dict,
    class_names: list[str],
    source_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = artifact["model"]
    if not hasattr(model, "coef_"):
        empty = pd.DataFrame(
            columns=[
                "class_name",
                "transformed_feature",
                "coefficient",
                "abs_coefficient",
                "interpretation",
            ]
        )
        return empty, empty

    feature_names = transformed_feature_names(artifact, source_df, feature_cols)
    if len(feature_names) != model.coef_.shape[1]:
        feature_names = [f"transformed_feature_{i}" for i in range(model.coef_.shape[1])]

    rows = []
    for class_idx, class_name in enumerate(class_names):
        coef_values = model.coef_[class_idx]
        for feature_name, coef in zip(feature_names, coef_values):
            rows.append(
                {
                    "class_name": class_name,
                    "transformed_feature": str(feature_name),
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                    "interpretation": (
                        f"Sube la probabilidad relativa de {class_name}"
                        if coef > 0
                        else f"Baja la probabilidad relativa de {class_name}"
                    ),
                }
            )

    params = pd.DataFrame(rows).sort_values(
        ["class_name", "abs_coefficient"], ascending=[True, False]
    )
    top_rows = []
    for class_name, gdf in params.groupby("class_name"):
        pos = gdf.sort_values("coefficient", ascending=False).head(8).assign(direction="empuja_clase")
        neg = gdf.sort_values("coefficient", ascending=True).head(8).assign(direction="aleja_clase")
        top_rows.append(pd.concat([pos, neg], ignore_index=True))
    top = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    return params, top


def plot_confusion_matrix(pred_rows: pd.DataFrame, class_names: list[str]) -> None:
    matrix = pd.crosstab(
        pred_rows["actual_class"],
        pred_rows["predicted_class"],
        rownames=["Real"],
        colnames=["Predicho"],
    ).reindex(index=class_names, columns=class_names, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Matriz de confusion en test")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_confusion_matrix_heatmap.png", dpi=160)
    plt.close(fig)


def plot_risk_distribution(latest: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(
        data=latest,
        x="priority_score_0_100",
        hue="actual_class",
        bins=18,
        multiple="stack",
        ax=ax,
    )
    ax.set_title("Distribucion del score de prioridad en test")
    ax.set_xlabel("Score ponderado: +30, +60 y +90")
    ax.set_ylabel("Facturas")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_test_risk_distribution.png", dpi=160)
    plt.close(fig)


def plot_new_scenarios(scenarios_pred: pd.DataFrame, class_names: list[str]) -> None:
    proba_cols = [f"prob_{label_slug(c)}" for c in class_names]
    plot_df = scenarios_pred[["escenario", *proba_cols]].copy()
    plot_df = plot_df.melt(
        id_vars="escenario",
        var_name="clase",
        value_name="probabilidad",
    )
    plot_df["clase"] = plot_df["clase"].str.replace("prob_", "", regex=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="probabilidad", y="escenario", hue="clase", ax=ax)
    ax.set_title("Probabilidades por clase en facturas nuevas")
    ax.set_xlabel("Probabilidad")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.legend(title="Clase", loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_new_scenarios_probabilities.png", dpi=160)
    plt.close(fig)


def plot_timeline_examples(timeline: pd.DataFrame) -> None:
    if timeline.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for factura_id, gdf in timeline.groupby(ID_COL):
        label = f"{factura_id} - {gdf['example_type'].iloc[0]}"
        ax.plot(
            pd.to_datetime(gdf[DATE_COL]),
            gdf["priority_score_0_100"],
            marker="o",
            label=label,
        )
    ax.set_title("Evolucion del riesgo predicho por cortes")
    ax.set_xlabel("Fecha de corte")
    ax.set_ylabel("Score prioridad 0-100")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_invoice_timeline_examples.png", dpi=160)
    plt.close(fig)


def main() -> None:
    require_files([DATA_PATH, TEST_IDS_PATH, MODEL_PATH, FEATURE_SCHEMA_PATH, METADATA_PATH])

    df = pd.read_csv(DATA_PATH)
    test_ids = pd.read_csv(TEST_IDS_PATH)[ID_COL].astype(str).str.strip()
    artifact = joblib.load(MODEL_PATH)
    feature_cols = pd.read_csv(FEATURE_SCHEMA_PATH)["feature"].dropna().astype(str).tolist()

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    artifact["feature_cols"] = feature_cols
    class_names = class_names_from_model(artifact["model"], artifact.get("target_encoder"))
    if not class_names:
        class_names = [str(c) for c in metadata.get("class_names", [])]

    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df_test = df[df[ID_COL].isin(set(test_ids.tolist()))].copy()
    if df_test.empty:
        sys.exit("[ERROR] El dataset de test quedo vacio.")

    pred_rows = add_prediction_columns(df_test, artifact, class_names)
    visible_cols = [
        ID_COL,
        CLIENT_COL,
        DATE_COL,
        "monto",
        "dias_hasta_vence",
        "dias_mora_observable",
        "num_gestiones_factura",
        "tasa_cumplimiento",
        "tasa_contacto_cliente",
        "num_promesas_rotas",
        "actual_class",
        "predicted_class",
        "is_correct",
        "confidence_probability",
        "high_risk_probability",
        "any_late_probability",
        "priority_score_0_100",
    ]
    proba_cols = [f"prob_{label_slug(c)}" for c in class_names]
    export_cols = [c for c in [*visible_cols, *proba_cols] if c in pred_rows.columns]
    pred_rows[export_cols].to_csv(PROBA_ROWS_PATH, index=False)

    latest = latest_cut_per_invoice(pred_rows)
    latest_export_cols = [
        c
        for c in [
            ID_COL,
            CLIENT_COL,
            DATE_COL,
            "monto",
            "dias_hasta_vence",
            "dias_mora_observable",
            "num_gestiones_factura",
            "tasa_cumplimiento",
            "tasa_contacto_cliente",
            "num_promesas_rotas",
            "actual_class",
            "predicted_class",
            "is_correct",
            "confidence_probability",
            "high_risk_probability",
            "any_late_probability",
            "priority_score_0_100",
            *proba_cols,
        ]
        if c in latest.columns
    ]
    latest[latest_export_cols].to_csv(INVOICE_SUMMARY_PATH, index=False)

    timeline = choose_timeline_examples(pred_rows, latest)
    timeline_export_cols = [
        c
        for c in [
            "example_type",
            ID_COL,
            CLIENT_COL,
            DATE_COL,
            "monto",
            "dias_hasta_vence",
            "dias_mora_observable",
            "num_gestiones_factura",
            "actual_class",
            "predicted_class",
            "confidence_probability",
            "priority_score_0_100",
            *proba_cols,
        ]
        if c in timeline.columns
    ]
    timeline[timeline_export_cols].to_csv(TIMELINE_EXAMPLES_PATH, index=False)

    scenarios = build_new_invoice_scenarios(df, feature_cols)
    scenario_input_cols = [
        c
        for c in [
            "escenario",
            "lectura_esperada",
            ID_COL,
            CLIENT_COL,
            DATE_COL,
            *feature_cols,
        ]
        if c in scenarios.columns
    ]
    scenarios[scenario_input_cols].to_csv(NEW_SCENARIO_INPUTS_PATH, index=False)

    scenario_pred = add_prediction_columns(scenarios, artifact, class_names)
    scenario_export_cols = [
        c
        for c in [
            "escenario",
            "lectura_esperada",
            ID_COL,
            CLIENT_COL,
            DATE_COL,
            "monto",
            "dias_hasta_vence",
            "dias_mora_observable",
            "num_gestiones_factura",
            "tasa_cumplimiento",
            "tasa_contacto_cliente",
            "num_promesas_rotas",
            "predicted_class",
            "confidence_probability",
            "high_risk_probability",
            "any_late_probability",
            "priority_score_0_100",
            *proba_cols,
        ]
        if c in scenario_pred.columns
    ]
    scenario_pred[scenario_export_cols].to_csv(NEW_SCENARIOS_PATH, index=False)

    feature_dictionary = build_feature_dictionary(feature_cols)
    feature_dictionary.to_csv(FEATURE_DICTIONARY_PATH, index=False)

    params, top_params = export_model_parameters(artifact, class_names, df, feature_cols)
    params.to_csv(MODEL_PARAMETERS_PATH, index=False)
    top_params.to_csv(TOP_PARAMETERS_PATH, index=False)

    plot_confusion_matrix(pred_rows, class_names)
    plot_risk_distribution(latest)
    plot_timeline_examples(timeline)
    plot_new_scenarios(scenario_pred, class_names)

    accuracy_latest = latest["is_correct"].mean()
    print("=" * 70)
    print("EXPLORACION DE PREDICCIONES - FASE 4")
    print("=" * 70)
    print(f"Modelo: {metadata.get('best_model', artifact.get('model_name', 'desconocido'))}")
    print(f"Filas de test evaluadas: {len(pred_rows):,}")
    print(f"Facturas test evaluadas en ultimo corte: {len(latest):,}")
    print(f"Accuracy en ultimo corte por factura: {accuracy_latest:.4f}")
    print("\nFacturas nuevas de ejemplo:")
    print(
        scenario_pred[
            [
                "escenario",
                "predicted_class",
                "confidence_probability",
                "priority_score_0_100",
            ]
        ].to_string(index=False)
    )
    print("\n[SAVED] Outputs de prediccion guardados en:")
    for path in [
        PROBA_ROWS_PATH,
        INVOICE_SUMMARY_PATH,
        TIMELINE_EXAMPLES_PATH,
        NEW_SCENARIOS_PATH,
        FEATURE_DICTIONARY_PATH,
        TOP_PARAMETERS_PATH,
    ]:
        print(f"  - {path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
