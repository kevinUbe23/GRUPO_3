# =============================================================================
# model_eval_cobranzas.py  —  Versión final robusta y defendible académicamente
# Componente 1 — Evaluación de modelos para priorización de cobranzas
# Proyecto: GRUPO_3
# Ruta destino: GRUPO_3/Evaluación modelo IA/
# Salidas:    GRUPO_3/Evaluación modelo IA/data/
# =============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ── XGBoost opcional ─────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost no disponible — se omitira.")

# =============================================================================
# RUTAS ROBUSTAS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PREP_DATA_DIR = PROJECT_DIR / "Presentación de la fase de preparación y procesamiento de datos" / "data"
OUTPUT_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV      = PREP_DATA_DIR / "features_ml_prepared.csv"
TRAIN_IDS_CSV  = PREP_DATA_DIR / "train_facturas_ids.csv"
TEST_IDS_CSV   = PREP_DATA_DIR / "test_facturas_ids.csv"

# =============================================================================
# CONSTANTES CONFIGURABLES
# =============================================================================
TARGET_COL         = "target_mora"
ID_COL             = "factura_id"
CLIENT_COL         = "cliente_id"
SECTOR_OHE_PREFIX  = "sector_"
SPLIT_TEST_SIZE    = 0.20
SPLIT_SEED         = 42
MIN_FAIRNESS_GROUP = 30     # tamano minimo de grupo para analisis de fairness
MAX_SELECTION_GAP  = 0.10   # gap maximo aceptable para escoger modelo

# Columnas de identificacion / leakage — NUNCA predictores
ID_COLS_EXCLUDE = [
    "factura_id", "cliente_id", "corte_id",
    "fecha_emision", "fecha_vencimiento", "fecha_corte",
    "periodo", "fecha_registro", "fecha_ultimo_pago",
    "fecha_primer_mora", "fecha_pago_real",
]

# Variables de negocio sesgadas — transformacion log1p
LOG1P_COLS = [
    "monto", "monto_promedio_hist", "ratio_monto",
    "mora_promedio_hist", "mora_ultimo_tramo",
    "num_gestiones_factura", "dias_hasta_vence_pos",
    "dias_mora_observable", "num_no_contesta_cons",
    "num_promesas_rotas", "promesas_total",
    "dias_transcurridos_corte",
]

# Columnas de fairness a evaluar (si existen en el dataset)
FAIRNESS_COLS = ["sector_dominante", "tiene_garantia", "tiene_disputa_activa"]

# Universo controlado de features permitidas para evitar leakage accidental
APPROVED_FEATURE_COLS = [
    "monto", "condicion_dias", "antiguedad_meses", "tiene_garantia",
    "sector_retail", "sector_manufactura", "sector_servicios",
    "sector_construccion", "sector_agro", "sector_tecnologia",
    "sector_salud", "sector_transporte", "num_facturas_prev",
    "mora_promedio_hist", "mora_ultimo_tramo", "tasa_cumplimiento",
    "monto_promedio_hist", "ratio_monto", "moras_consecutivas",
    "num_gestiones_factura", "dias_desde_ultima_gestion",
    "dias_hasta_vence", "tasa_contacto_cliente", "ultimo_resultado_enc",
    "num_no_contesta_cons", "tiene_disputa_activa",
    "tiene_promesa_activa", "num_promesas_rotas",
    "tasa_cumpl_promesas", "promesas_total", "sin_gestion_previa",
    "dias_transcurridos_corte", "esta_vencida_al_corte",
    "dias_mora_observable", "dias_hasta_vence_pos", "cliente_nuevo",
    "intensidad_gestion", "friccion_contacto", "ratio_promesas_rotas",
    "sector_dominante",
]

# =============================================================================
# HELPERS
# =============================================================================

def safe_auc(y_true, y_prob, n_classes):
    """Calcula ROC-AUC de forma segura para binario y multiclase."""
    try:
        if n_classes == 2:
            return roc_auc_score(y_true, y_prob[:, 1])
        else:
            return roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )
    except Exception as e:
        print(f"    [WARN] AUC no calculable: {e}")
        return np.nan


def metrics_for_split(y_true, y_pred, y_prob, n_classes):
    """Devuelve dict con todas las metricas para un split."""
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro":   precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":      recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":          f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_weighted":      safe_auc(y_true, y_prob, n_classes) if y_prob is not None else np.nan,
    }


def build_sector_dominante_from_ohe(df_in: pd.DataFrame, sector_cols: list[str]) -> pd.Series:
    """Devuelve sector dominante solo si exactamente una columna sector esta activa."""
    sector_matrix = df_in[sector_cols].fillna(0)
    active_counts = sector_matrix.gt(0).sum(axis=1)
    labels = sector_matrix.idxmax(axis=1).str.replace(SECTOR_OHE_PREFIX, "", regex=False)
    return labels.where(active_counts == 1, np.nan)


def worst_class_recall(y_true, y_pred, class_labels, class_name_map):
    """Calcula el peor recall por clase dentro de un grupo."""
    recalls = recall_score(
        y_true,
        y_pred,
        average=None,
        labels=class_labels,
        zero_division=0,
    )
    worst_idx = int(np.argmin(recalls))
    worst_label = class_labels[worst_idx]
    return float(recalls[worst_idx]), class_name_map.get(worst_label, str(worst_label))


def gap_diagnosis(gap_val: float) -> str:
    """Clasifica la magnitud del gap train-test."""
    if np.isnan(gap_val):
        return "n/a"
    if gap_val > 0.10:
        return "possible_overfitting"
    elif gap_val < -0.05:
        return "possible_underfitting"
    else:
        return "acceptable_gap"


def is_binary_col(series: pd.Series) -> bool:
    """Devuelve True si la columna contiene solo valores {0, 1}."""
    vals = set(series.dropna().unique())
    return vals.issubset({0, 1, True, False, 0.0, 1.0})


def make_ohe() -> OneHotEncoder:
    """Crea un OneHotEncoder compatible con distintas versiones de scikit-learn.
    sklearn >= 1.2 usa sparse_output=False; versiones anteriores usan sparse=False."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# =============================================================================
# 1. CARGA Y VALIDACIONES BASICAS
# =============================================================================
print("=" * 65)
print("COMPONENTE 1 — EVALUACION DE MODELOS DE COBRANZAS")
print("=" * 65)

missing_inputs = [p for p in [INPUT_CSV, TRAIN_IDS_CSV, TEST_IDS_CSV] if not p.exists()]
if missing_inputs:
    missing_txt = "\n".join(f"  - {p}" for p in missing_inputs)
    sys.exit(
        "\n[ERROR FATAL] Faltan archivos oficiales exportados desde preparacion:\n"
        f"{missing_txt}\n"
    )

df = pd.read_csv(INPUT_CSV)
print(f"[OK] features_ml_prepared.csv cargado: {df.shape}")

# Validar columnas obligatorias
for col in [TARGET_COL, ID_COL]:
    if col not in df.columns:
        sys.exit(
            f"\n[ERROR FATAL] Columna obligatoria '{col}' no encontrada.\n"
            f"Columnas disponibles: {list(df.columns)}\n"
        )

# =============================================================================
# 2. PREPARAR TARGET
# =============================================================================
n_before = len(df)
df = df.dropna(subset=[TARGET_COL])
print(f"[INFO] Filas eliminadas por NaN en target: {n_before - len(df)}")
df[ID_COL] = df[ID_COL].astype(str).str.strip()

target_counts_by_factura = df.groupby(ID_COL)[TARGET_COL].nunique(dropna=False)
inconsistent_facturas = target_counts_by_factura[target_counts_by_factura > 1].index.tolist()
if inconsistent_facturas:
    sample_ids = ", ".join(map(str, inconsistent_facturas[:5]))
    sys.exit(
        f"\n[ERROR FATAL] Se detectaron {len(inconsistent_facturas)} facturas con target inconsistente.\n"
        f"Ejemplos: {sample_ids}\n"
        "Revisa el dataset antes de evaluar.\n"
    )

# Codificar target si es string/category
target_le = None
if (
    pd.api.types.is_object_dtype(df[TARGET_COL])
    or pd.api.types.is_categorical_dtype(df[TARGET_COL])
    or pd.api.types.is_string_dtype(df[TARGET_COL])
):
    target_le = LabelEncoder()
    df[TARGET_COL] = target_le.fit_transform(df[TARGET_COL].astype(str))
    print(f"[OK] Target codificado con LabelEncoder. Clases: {list(target_le.classes_)}")

n_classes = int(df[TARGET_COL].nunique())
print(f"[OK] Clases en target: {n_classes} — {sorted(df[TARGET_COL].unique())}")

# =============================================================================
# 3. CONSTRUIR sector_dominante SI NO EXISTE
# =============================================================================
sector_oh_cols = [
    c for c in df.columns
    if c.startswith(SECTOR_OHE_PREFIX)
    and c not in ID_COLS_EXCLUDE
    and c != TARGET_COL
]

if "sector_dominante" not in df.columns and sector_oh_cols:
    df["sector_dominante"] = build_sector_dominante_from_ohe(df, sector_oh_cols)
    print(f"[OK] 'sector_dominante' creada desde {len(sector_oh_cols)} columnas one-hot.")

# =============================================================================
# 4. SPLIT OFICIAL POR FACTURA_ID
# =============================================================================
train_ids_raw = pd.read_csv(TRAIN_IDS_CSV)[ID_COL].astype(str).str.strip()
test_ids_raw  = pd.read_csv(TEST_IDS_CSV)[ID_COL].astype(str).str.strip()

train_dup = int(train_ids_raw.duplicated().sum())
test_dup  = int(test_ids_raw.duplicated().sum())
if train_dup or test_dup:
    sys.exit(
        f"\n[ERROR FATAL] IDs duplicados en archivos oficiales de split. "
        f"train_dup={train_dup}, test_dup={test_dup}\n"
    )

train_ids = set(train_ids_raw.tolist())
test_ids  = set(test_ids_raw.tolist())
dataset_ids = set(df[ID_COL].dropna().unique())
print(f"[OK] IDs de split oficiales cargados desde preparacion.")

# Validar sin interseccion
overlap = train_ids & test_ids
if overlap:
    sys.exit(
        f"\n[ERROR FATAL] {len(overlap)} facturas aparecen en train Y test.\n"
        "Revisa los archivos oficiales de split antes de ejecutar.\n"
    )

missing_ids = dataset_ids - (train_ids | test_ids)
if missing_ids:
    sample_ids = ", ".join(sorted(list(missing_ids))[:5])
    sys.exit(
        f"\n[ERROR FATAL] Hay {len(missing_ids)} facturas del dataset que no estan en train ni test.\n"
        f"Ejemplos: {sample_ids}\n"
    )

extra_ids = (train_ids | test_ids) - dataset_ids
if extra_ids:
    sample_ids = ", ".join(sorted(list(extra_ids))[:5])
    sys.exit(
        f"\n[ERROR FATAL] Hay {len(extra_ids)} facturas en los archivos de split que no existen en el dataset.\n"
        f"Ejemplos: {sample_ids}\n"
    )

df_train = df[df[ID_COL].isin(train_ids)].copy()
df_test  = df[df[ID_COL].isin(test_ids)].copy()

if df_train.empty or df_test.empty:
    sys.exit("[ERROR FATAL] Train o test quedaron vacios tras el split.")

print(f"  Filas train:    {len(df_train):,}   |  Filas test:    {len(df_test):,}")
print(f"  Facturas train: {df_train[ID_COL].nunique():,}  |  Facturas test: {df_test[ID_COL].nunique():,}")

y_train = df_train[TARGET_COL].to_numpy()
y_test  = df_test[TARGET_COL].to_numpy()

# =============================================================================
# 5. CLASIFICAR COLUMNAS DE FEATURES
# =============================================================================
exclude_always = list(set(
    [c for c in ID_COLS_EXCLUDE if c in df.columns] + [TARGET_COL]
))

# Columnas categoricas nominales — OHE (NO LabelEncoder automatico)
nominal_cols = [
    c for c in ["ultimo_resultado_enc", "sector_dominante"]
    if c in df.columns and c not in exclude_always
]

# Columnas log1p: existen, son numericas, no son nominales ni excluidas
log1p_present = [
    c for c in LOG1P_COLS
    if c in df.columns
    and c not in exclude_always
    and c not in nominal_cols
    and pd.api.types.is_numeric_dtype(df[c])
]

# Columnas binarias
binary_cols = [
    c for c in df.columns
    if c not in exclude_always
    and c not in nominal_cols
    and c not in log1p_present
    and pd.api.types.is_numeric_dtype(df[c])
    and is_binary_col(df[c])
]

# Numericas restantes
numeric_other = [
    c for c in df.columns
    if c not in exclude_always
    and c not in nominal_cols
    and c not in log1p_present
    and c not in binary_cols
    and pd.api.types.is_numeric_dtype(df[c])
]

feature_cols = log1p_present + numeric_other + binary_cols + nominal_cols

# Endurecer el esquema de features para evitar leakage por columnas nuevas.
approved_feature_cols = [c for c in APPROVED_FEATURE_COLS if c in feature_cols]
unexpected_numeric = sorted([
    c for c in feature_cols
    if c not in approved_feature_cols and pd.api.types.is_numeric_dtype(df[c])
])
if unexpected_numeric:
    print(
        f"[WARN] {len(unexpected_numeric)} columnas numericas no aprobadas fueron excluidas: "
        f"{unexpected_numeric}"
    )

nominal_cols = [
    c for c in ["ultimo_resultado_enc", "sector_dominante"]
    if c in approved_feature_cols
]
log1p_present = [
    c for c in LOG1P_COLS
    if c in approved_feature_cols
    and c not in nominal_cols
    and pd.api.types.is_numeric_dtype(df[c])
]
binary_cols = [
    c for c in approved_feature_cols
    if c not in nominal_cols
    and c not in log1p_present
    and pd.api.types.is_numeric_dtype(df[c])
    and is_binary_col(df[c])
]
numeric_other = [
    c for c in approved_feature_cols
    if c not in nominal_cols
    and c not in log1p_present
    and c not in binary_cols
    and pd.api.types.is_numeric_dtype(df[c])
]
feature_cols = log1p_present + numeric_other + binary_cols + nominal_cols

# Eliminar columnas con varianza cero en train
zero_var = [
    c for c in (log1p_present + numeric_other + binary_cols)
    if df_train[c].var() == 0
]
if zero_var:
    print(f"[INFO] {len(zero_var)} columnas con varianza cero excluidas: {zero_var}")
    feature_cols   = [c for c in feature_cols   if c not in zero_var]
    log1p_present  = [c for c in log1p_present  if c not in zero_var]
    numeric_other  = [c for c in numeric_other  if c not in zero_var]
    binary_cols    = [c for c in binary_cols    if c not in zero_var]

print(f"\n[OK] Distribucion de features:")
print(f"  log1p:    {len(log1p_present)}")
print(f"  numeric:  {len(numeric_other)}")
print(f"  binary:   {len(binary_cols)}")
print(f"  nominal:  {len(nominal_cols)}")
print(f"  TOTAL:    {len(feature_cols)}")

# =============================================================================
# 6. PREPROCESSORS
# =============================================================================

# ── Transformador log1p con escalado (para LR) ──────────────────────────────
log1p_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p",   FunctionTransformer(np.log1p, validate=True)),
    ("scaler",  StandardScaler()),
])
numeric_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
binary_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])
nominal_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     make_ohe()),
])

preprocessor_scaled = ColumnTransformer(
    transformers=[
        ("log1p",   log1p_scaled,   log1p_present),
        ("numeric", numeric_scaled, numeric_other),
        ("binary",  binary_pipe,    binary_cols),
        ("nominal", nominal_pipe,   nominal_cols),
    ],
    remainder="drop"
)

# ── Transformador sin escalado (para arboles) ────────────────────────────────
log1p_noscale = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p",   FunctionTransformer(np.log1p, validate=True)),
])
numeric_noscale = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

preprocessor_tree = ColumnTransformer(
    transformers=[
        ("log1p",   log1p_noscale,  log1p_present),
        ("numeric", numeric_noscale, numeric_other),
        ("binary",  binary_pipe,    binary_cols),
        ("nominal", nominal_pipe,   nominal_cols),
    ],
    remainder="drop"
)

X_train_df = df_train[feature_cols]
X_test_df  = df_test[feature_cols]

X_train_sc   = preprocessor_scaled.fit_transform(X_train_df)
X_test_sc    = preprocessor_scaled.transform(X_test_df)
X_train_tree = preprocessor_tree.fit_transform(X_train_df)
X_test_tree  = preprocessor_tree.transform(X_test_df)

print(f"[OK] Preprocessors aplicados. "
      f"Escalado: {X_train_sc.shape} | Arbol: {X_train_tree.shape}")

# =============================================================================
# 7. MANEJO XGBOOST (enteros desde 0, decodificacion posterior)
# =============================================================================
xgb_le       = None
y_train_xgb  = None
y_test_xgb   = None

if XGBOOST_AVAILABLE:
    xgb_le      = LabelEncoder()
    y_train_xgb = xgb_le.fit_transform(y_train)
    y_test_xgb  = xgb_le.transform(y_test)

# =============================================================================
# 8. DEFINICION DE MODELOS
# =============================================================================
MODELS = {
    "Dummy Baseline": {
        "model": DummyClassifier(strategy="most_frequent", random_state=SPLIT_SEED),
        "X_tr": X_train_sc,
        "X_te": X_test_sc,
        "y_tr": y_train,
        "y_te": y_test,
        "is_xgb": False,
    },
    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=1000, random_state=SPLIT_SEED,
            class_weight="balanced", solver="lbfgs"
        ),
        "X_tr": X_train_sc,
        "X_te": X_test_sc,
        "y_tr": y_train,
        "y_te": y_test,
        "is_xgb": False,
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=200, random_state=SPLIT_SEED,
            class_weight="balanced", n_jobs=-1
        ),
        "X_tr": X_train_tree,
        "X_te": X_test_tree,
        "y_tr": y_train,
        "y_te": y_test,
        "is_xgb": False,
    },
}

if XGBOOST_AVAILABLE:
    xgb_params = dict(
        n_estimators=200,
        random_state=SPLIT_SEED,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
    )
    if n_classes > 2:
        xgb_params["objective"]  = "multi:softprob"
        xgb_params["num_class"]  = n_classes
    MODELS["XGBoost"] = {
        "model": XGBClassifier(**xgb_params),
        "X_tr": X_train_tree,
        "X_te": X_test_tree,
        "y_tr": y_train_xgb,
        "y_te": y_test_xgb,
        "is_xgb": True,
    }

# =============================================================================
# 9. ENTRENAMIENTO Y METRICAS — formato wide (una fila por modelo)
# =============================================================================
benchmark_rows  = []
trained_results = {}

print("\n── ENTRENAMIENTO ────────────────────────────────────────────────")

for name, cfg in MODELS.items():
    model   = cfg["model"]
    X_tr    = cfg["X_tr"]
    X_te    = cfg["X_te"]
    y_tr    = cfg["y_tr"]
    y_te    = cfg["y_te"]
    is_xgb  = cfg["is_xgb"]

    print(f"\n  [{name}] entrenando ...")
    try:
        model.fit(X_tr, y_tr)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        continue

    # Predicciones
    y_pred_tr_raw = model.predict(X_tr)
    y_pred_te_raw = model.predict(X_te)
    y_prob_tr     = model.predict_proba(X_tr) if hasattr(model, "predict_proba") else None
    y_prob_te     = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None

    # Decodificar XGBoost al espacio de etiquetas original
    if is_xgb and xgb_le is not None:
        y_pred_tr = xgb_le.inverse_transform(y_pred_tr_raw)
        y_pred_te = xgb_le.inverse_transform(y_pred_te_raw)
        y_true_tr = y_train
        y_true_te = y_test
    else:
        y_pred_tr = y_pred_tr_raw
        y_pred_te = y_pred_te_raw
        y_true_tr = y_tr
        y_true_te = y_te

    m_tr = metrics_for_split(y_true_tr, y_pred_tr, y_prob_tr, n_classes)
    m_te = metrics_for_split(y_true_te, y_pred_te, y_prob_te, n_classes)

    row = {"model": name}
    for k, v in m_tr.items():
        row[f"{k}_train"] = round(v, 6) if not np.isnan(v) else np.nan
    for k, v in m_te.items():
        row[f"{k}_test"]  = round(v, 6) if not np.isnan(v) else np.nan
    benchmark_rows.append(row)

    trained_results[name] = {
        "model":     model,
        "X_te":      X_te,
        "y_true_te": y_true_te,
        "y_pred_te": y_pred_te,
        "y_prob_te": y_prob_te,
    }
    print(f"  [OK] f1_train={m_tr['f1_macro']:.4f} | f1_test={m_te['f1_macro']:.4f}")

benchmark_df = pd.DataFrame(benchmark_rows)
benchmark_path = OUTPUT_DIR / "benchmark_metrics.csv"
benchmark_df.to_csv(benchmark_path, index=False)
print(f"\n[SAVED] {benchmark_path}")
print(benchmark_df.to_string(index=False))

# =============================================================================
# 10. TRAIN-TEST GAP CON DIAGNOSTICO
# =============================================================================
gap_rows = []
for _, row in benchmark_df.iterrows():
    name    = row["model"]
    acc_tr  = row.get("accuracy_train",     np.nan)
    acc_te  = row.get("accuracy_test",      np.nan)
    f1_tr   = row.get("f1_macro_train",     np.nan)
    f1_te   = row.get("f1_macro_test",      np.nan)
    auc_tr  = row.get("auc_weighted_train", np.nan)
    auc_te  = row.get("auc_weighted_test",  np.nan)

    gap_acc = (acc_tr - acc_te) if not (np.isnan(acc_tr) or np.isnan(acc_te)) else np.nan
    gap_f1  = (f1_tr  - f1_te)  if not (np.isnan(f1_tr)  or np.isnan(f1_te))  else np.nan
    gap_auc = (auc_tr - auc_te) if not (np.isnan(auc_tr) or np.isnan(auc_te)) else np.nan

    diag_f1   = gap_diagnosis(gap_f1)
    diag_auc  = gap_diagnosis(gap_auc)
    diagnosis = diag_f1 if diag_f1 != "acceptable_gap" else diag_auc

    gap_rows.append({
        "model":          name,
        "accuracy_train": acc_tr,
        "accuracy_test":  acc_te,
        "f1_macro_train": f1_tr,
        "f1_macro_test":  f1_te,
        "auc_train":      auc_tr,
        "auc_test":       auc_te,
        "gap_accuracy":   gap_acc,
        "gap_f1_macro":   gap_f1,
        "gap_auc":        gap_auc,
        "diagnosis":      diagnosis,
    })

gap_df = pd.DataFrame(gap_rows)
gap_path = OUTPUT_DIR / "train_test_gap.csv"
gap_df.to_csv(gap_path, index=False)
print(f"[SAVED] {gap_path}")

# =============================================================================
# 11. SELECCION DEL MEJOR MODELO (prioriza F1 test con estabilidad)
# =============================================================================
non_baseline = benchmark_df[benchmark_df["model"] != "Dummy Baseline"].copy()
best_model_name = None
selection_reason = "sin candidatos validos"
if not non_baseline.empty and "f1_macro_test" in non_baseline.columns:
    candidate_df = non_baseline.merge(
        gap_df[["model", "gap_f1_macro", "gap_auc"]],
        on="model",
        how="left"
    )
    stable_df = candidate_df[
        candidate_df["gap_f1_macro"].isna()
        | (candidate_df["gap_f1_macro"] <= MAX_SELECTION_GAP)
    ].copy()
    ranking_cols = ["f1_macro_test", "balanced_accuracy_test", "auc_weighted_test", "gap_f1_macro"]
    ranking_asc = [False, False, False, True]
    if not stable_df.empty:
        stable_df = stable_df.sort_values(ranking_cols, ascending=ranking_asc)
        best_model_name = stable_df.iloc[0]["model"]
        selection_reason = (
            f"mayor F1-macro test entre modelos con gap_f1 <= {MAX_SELECTION_GAP:.2f}"
        )
    else:
        candidate_df = candidate_df.sort_values(ranking_cols, ascending=ranking_asc)
        best_model_name = candidate_df.iloc[0]["model"]
        selection_reason = (
            "fallback al mayor F1-macro test porque ningun modelo cumplio el criterio de estabilidad"
        )
        print(f"[WARN] {selection_reason}.")
print(f"\n[OK] Mejor modelo seleccionado: {best_model_name}")

# =============================================================================
# 12. CLASSIFICATION REPORT DEL MEJOR MODELO
# =============================================================================
report_path = OUTPUT_DIR / "classification_report_best_model.txt"
confusion_matrix_path = OUTPUT_DIR / "confusion_matrix_best_model.csv"
class_metrics_path = OUTPUT_DIR / "class_metrics_best_model.csv"
if best_model_name and best_model_name in trained_results:
    res    = trained_results[best_model_name]
    if target_le is not None:
        class_labels = list(range(n_classes))
        class_names = [str(c) for c in target_le.classes_]
    else:
        class_labels = sorted(pd.Series(res["y_true_te"]).dropna().unique().tolist())
        class_names = [str(c) for c in class_labels]
    report = classification_report(
        res["y_true_te"],
        res["y_pred_te"],
        labels=class_labels,
        target_names=class_names,
        zero_division=0
    )
    report_dict = classification_report(
        res["y_true_te"],
        res["y_pred_te"],
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(res["y_true_te"], res["y_pred_te"], labels=class_labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = "actual_class"
    cm_df.columns.name = "predicted_class"
    class_metrics_df = (
        pd.DataFrame(report_dict)
        .T
        .reset_index()
        .rename(columns={"index": "class_name"})
    )
    class_metrics_df = class_metrics_df[
        ~class_metrics_df["class_name"].isin(["accuracy", "macro avg", "weighted avg"])
    ].copy()
    class_metrics_df["class_label"] = class_labels
    class_metrics_df = class_metrics_df[
        ["class_label", "class_name", "precision", "recall", "f1-score", "support"]
    ]
    print(f"\n── Classification Report [{best_model_name}] ──")
    print(report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Modelo: {best_model_name}\n\n{report}")
    cm_df.to_csv(confusion_matrix_path)
    class_metrics_df.to_csv(class_metrics_path, index=False)
    print(f"[SAVED] {report_path}")
    print(f"[SAVED] {confusion_matrix_path}")
    print(f"[SAVED] {class_metrics_path}")

# =============================================================================
# 13. FAIRNESS ANALYSIS
# =============================================================================
# Construir sector_dominante en df_test si falta
if "sector_dominante" not in df_test.columns and sector_oh_cols:
    df_test["sector_dominante"] = build_sector_dominante_from_ohe(df_test, sector_oh_cols)

fairness_present = [c for c in FAIRNESS_COLS if c in df_test.columns]
print(f"\n[FAIRNESS] Columnas disponibles: {fairness_present}")
print(f"[FAIRNESS] Tamano minimo de grupo: {MIN_FAIRNESS_GROUP}")

fairness_rows    = []
gap_summary_rows = []

if best_model_name and best_model_name in trained_results:
    res = trained_results[best_model_name]
    df_eval = df_test.copy().reset_index(drop=True)
    df_eval["_y_pred"] = res["y_pred_te"]
    df_eval["_y_true"] = res["y_true_te"]
    class_labels = list(range(n_classes))
    class_name_map = {label: str(label) for label in class_labels}
    if target_le is not None:
        class_name_map = {label: target_le.classes_[label] for label in class_labels}

    for fc in fairness_present:
        for group_val, gdf in df_eval.groupby(fc):
            if len(gdf) < MIN_FAIRNESS_GROUP:
                continue
            g_true = gdf["_y_true"].values
            g_pred = gdf["_y_pred"].values
            worst_recall_value, worst_recall_class = worst_class_recall(
                g_true, g_pred, class_labels, class_name_map
            )
            fairness_rows.append({
                "model":             best_model_name,
                "fairness_col":      fc,
                "group":             group_val,
                "n":                 len(gdf),
                "accuracy":          accuracy_score(g_true, g_pred),
                "balanced_accuracy": balanced_accuracy_score(g_true, g_pred),
                "precision_macro":   precision_score(g_true, g_pred, average="macro", zero_division=0),
                "recall_macro":      recall_score(g_true, g_pred, average="macro", zero_division=0),
                "f1_macro":          f1_score(g_true, g_pred, average="macro", zero_division=0),
                "worst_class_recall": worst_recall_value,
                "worst_recall_class": worst_recall_class,
            })

        # Gap por columna
        fc_subset = [r for r in fairness_rows if r["fairness_col"] == fc]
        if len(fc_subset) < 2:
            continue
        fc_df = pd.DataFrame(fc_subset)
        for metric in ["accuracy", "balanced_accuracy", "f1_macro", "worst_class_recall"]:
            idx_max = fc_df[metric].idxmax()
            idx_min = fc_df[metric].idxmin()
            gap_summary_rows.append({
                "model":        best_model_name,
                "fairness_col": fc,
                "metric":       metric,
                "max_group":    fc_df.loc[idx_max, "group"],
                "max_value":    round(fc_df.loc[idx_max, metric], 6),
                "min_group":    fc_df.loc[idx_min, "group"],
                "min_value":    round(fc_df.loc[idx_min, metric], 6),
                "gap":          round(fc_df[metric].max() - fc_df[metric].min(), 6),
            })

fairness_df    = pd.DataFrame(fairness_rows)
gap_summary_df = pd.DataFrame(gap_summary_rows)

fairness_path    = OUTPUT_DIR / "fairness_by_group.csv"
gap_summary_path = OUTPUT_DIR / "fairness_gap_summary.csv"
fairness_df.to_csv(fairness_path, index=False)
gap_summary_df.to_csv(gap_summary_path, index=False)
print(f"[SAVED] {fairness_path}")
print(f"[SAVED] {gap_summary_path}")

if not fairness_df.empty:
    print(
        fairness_df[
            ["fairness_col", "group", "n", "f1_macro", "worst_class_recall", "worst_recall_class"]
        ].to_string(index=False)
    )
else:
    print("[INFO] No hay grupos con tamano suficiente para fairness analysis.")

# =============================================================================
# 14. SUMMARY.TXT COMPLETO Y DEFENDIBLE
# =============================================================================
best_bm_row  = benchmark_df[benchmark_df["model"] == best_model_name].iloc[0] \
               if best_model_name and not benchmark_df.empty else None
best_gap_row = gap_df[gap_df["model"] == best_model_name].iloc[0] \
               if best_model_name and not gap_df.empty else None

def fmt(val, dec=4):
    return f"{val:.{dec}f}" if not np.isnan(val) else "N/A"

lines = [
    "=" * 65,
    "RESUMEN EVALUACION — PRIORIZACION DE COBRANZAS (COMPONENTE 1)",
    "=" * 65,
    "",
    "── DATOS ──────────────────────────────────────────────────────",
    f"  Filas totales:           {len(df):,}",
    f"  Filas train:             {len(df_train):,}",
    f"  Filas test:              {len(df_test):,}",
    f"  Facturas train:          {df_train[ID_COL].nunique():,}",
    f"  Facturas test:           {df_test[ID_COL].nunique():,}",
    f"  Predictores finales:     {len(feature_cols)}",
    f"    - log1p:               {len(log1p_present)}",
    f"    - numericas:           {len(numeric_other)}",
    f"    - binarias:            {len(binary_cols)}",
    f"    - nominales (OHE):     {len(nominal_cols)}",
    f"  Clases en target:        {n_classes}",
    "",
    "── METRICAS TEST ───────────────────────────────────────────────",
]

for _, row in benchmark_df.iterrows():
    acc = fmt(row.get("accuracy_test",     np.nan))
    f1  = fmt(row.get("f1_macro_test",     np.nan))
    auc = fmt(row.get("auc_weighted_test", np.nan))
    bal = fmt(row.get("balanced_accuracy_test", np.nan))
    mark = "  <<< MEJOR" if row["model"] == best_model_name else ""
    lines.append(
        f"  {row['model']:<26} acc={acc} | bal_acc={bal} | f1={f1} | auc={auc}{mark}"
    )

lines += ["", "── GAPS OVERFITTING ─────────────────────────────────────────────"]
for _, row in gap_df.iterrows():
    lines.append(
        f"  {row['model']:<26} gap_f1={fmt(row['gap_f1_macro'])} "
        f"| gap_auc={fmt(row['gap_auc'])} | {row['diagnosis']}"
    )

if not gap_summary_df.empty:
    lines += ["", "── FAIRNESS GAPS ─────────────────────────────────────────────────"]
    for _, row in gap_summary_df.iterrows():
        lines.append(
            f"  {row['fairness_col']:<25} [{row['metric']}] "
            f"gap={fmt(row['gap'])} | "
            f"max={row['max_group']}({fmt(row['max_value'])}) "
            f"min={row['min_group']}({fmt(row['min_value'])})"
        )
else:
    lines += ["", "── FAIRNESS GAPS ─────────────────────────────────────────────────",
              "  Sin grupos suficientes para analisis de fairness."]

lines += ["", "── CONCLUSION ───────────────────────────────────────────────────"]
if best_model_name and best_bm_row is not None and best_gap_row is not None:
    f1_v   = best_bm_row.get("f1_macro_test",     np.nan)
    auc_v  = best_bm_row.get("auc_weighted_test", np.nan)
    diag   = best_gap_row["diagnosis"]
    if diag == "acceptable_gap":
        adj = "un ajuste generalizable y robusto"
    elif diag == "possible_underfitting":
        adj = "un modelo probablemente conservador o insuficientemente ajustado"
    else:
        adj = "posibles signos de sobreajuste"
    lines.append(
        f"  Modelo seleccionado: '{best_model_name}'\n"
        f"  F1-macro test: {fmt(f1_v)} | AUC-weighted test: {fmt(auc_v)}\n"
        f"  Diagnostico train-test: '{diag}', lo que sugiere {adj}.\n"
        f"  Criterio de seleccion: {selection_reason}.\n"
        f"  El split se leyo desde los archivos oficiales de preparacion por factura_id\n"
        f"  para garantizar que no haya fuga de datos entre train y test."
    )

lines.append("=" * 65)

summary_txt = "\n".join(lines)
print("\n" + summary_txt)

summary_path = OUTPUT_DIR / "summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_txt)
print(f"\n[SAVED] {summary_path}")

# =============================================================================
# VERIFICACION FINAL DE ARCHIVOS
# =============================================================================
print("\n" + "=" * 65)
print("ARCHIVOS GENERADOS:")
for fname in [
    "benchmark_metrics.csv",
    "train_test_gap.csv",
    "fairness_by_group.csv",
    "fairness_gap_summary.csv",
    "summary.txt",
    "classification_report_best_model.txt",
    "confusion_matrix_best_model.csv",
    "class_metrics_best_model.csv",
]:
    p = OUTPUT_DIR / fname
    status = "OK" if p.exists() else "FALTANTE"
    print(f"  [{status}] {p}")
print("=" * 65)
print("[OK] Componente 1 completado exitosamente.")
