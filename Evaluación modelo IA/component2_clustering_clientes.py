# =============================================================================
# component2_clustering_clientes.py — Version final robusta y defendible
# Componente 2 — Clustering de clientes para priorizacion de cobranzas
# Proyecto: GRUPO_3
# Ruta destino: GRUPO_3/Evaluacion modelo IA/
# Salidas:    GRUPO_3/Evaluacion modelo IA/data/
# =============================================================================

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")

# =============================================================================
# RUTAS ROBUSTAS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "features_ml_prepared.csv"

# =============================================================================
# CONSTANTES CONFIGURABLES
# =============================================================================
CLIENT_COL         = "cliente_id"
TARGET_COL         = "target_mora"          # excluido del clustering
ID_COL             = "factura_id"
SECTOR_PREFIX      = "sector_"
KMEANS_K_FIXED     = 5                      # k principal alineado al proyecto
KMEANS_K_SEARCH    = range(2, 9)            # busqueda auxiliar
KMEANS_SEED        = 42
DBSCAN_EPS         = 0.5
DBSCAN_MIN_SAMPLES = 5

# Columnas de identificacion y leakage — NUNCA en clustering
EXCLUDE_ALWAYS = [
    "factura_id", "corte_id", "cliente_id",
    "fecha_emision", "fecha_vencimiento", "fecha_corte",
    "periodo", "fecha_registro", "fecha_ultimo_pago",
    "fecha_primer_mora", "fecha_pago_real",
    "target_mora",          # componente NO supervisado
    "ultimo_resultado_enc", # nominal — no ordinalizar artificialmente
]

# Variables de comportamiento historico priorizadas para clustering
BEHAVIOR_VARS = [
    "monto",
    "condicion_dias",
    "antiguedad_meses",
    "num_facturas_prev",
    "mora_promedio_hist",
    "mora_ultimo_tramo",
    "tasa_cumplimiento",
    "monto_promedio_hist",
    "ratio_monto",
    "moras_consecutivas",
    "num_gestiones_factura",
    "dias_desde_ultima_gestion",
    "dias_hasta_vence",
    "tasa_contacto_cliente",
    "num_no_contesta_cons",
    "tiene_disputa_activa",
    "tiene_promesa_activa",
    "num_promesas_rotas",
    "tasa_cumpl_promesas",
    "promesas_total",
    "sin_gestion_previa",
    "esta_vencida_al_corte",
    "dias_mora_observable",
    "dias_hasta_vence_pos",
    "cliente_nuevo",
    "intensidad_gestion",
    "friccion_contacto",
    "ratio_promesas_rotas",
]

# Variables sesgadas que se benefician de log1p antes de agregar
LOG1P_VARS = [
    "monto",
    "monto_promedio_hist",
    "ratio_monto",
    "promesas_total",
    "num_promesas_rotas",
    "num_gestiones_factura",
    "dias_mora_observable",
    "num_no_contesta_cons",
    "dias_hasta_vence_pos",
    "dias_desde_ultima_gestion",
]

# Agregaciones logicas por tipo de variable
#   continuas:  mean, max
#   ratios:     mean
#   conteos:    sum, mean
#   binarias:   mean (tasa de ocurrencia)
CONT_VARS = [
    "monto", "condicion_dias", "antiguedad_meses",
    "mora_promedio_hist", "mora_ultimo_tramo",
    "monto_promedio_hist", "ratio_monto",
    "dias_desde_ultima_gestion", "dias_hasta_vence",
    "dias_mora_observable", "dias_hasta_vence_pos",
    "intensidad_gestion", "friccion_contacto",
]
RATIO_VARS = [
    "tasa_cumplimiento", "tasa_contacto_cliente",
    "tasa_cumpl_promesas", "ratio_promesas_rotas",
]
COUNT_VARS = [
    "num_facturas_prev", "moras_consecutivas",
    "num_gestiones_factura", "num_no_contesta_cons",
    "num_promesas_rotas", "promesas_total",
]
BINARY_VARS = [
    "tiene_disputa_activa", "tiene_promesa_activa",
    "sin_gestion_previa", "esta_vencida_al_corte", "cliente_nuevo",
]

# =============================================================================
# HELPERS
# =============================================================================

def safe_log1p(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Aplica log1p a columnas existentes y no negativas."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            # Clampear negativos a 0 antes de log1p (p.ej. dias_hasta_vence puede ser neg)
            df[c] = np.log1p(np.maximum(df[c].fillna(0), 0))
    return df


def cluster_metrics(X: np.ndarray, labels: np.ndarray, tag: str) -> dict:
    """Calcula silhouette, Davies-Bouldin y Calinski-Harabasz de forma segura."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_mask = labels == -1
    noise_n    = int(noise_mask.sum())
    noise_ratio = noise_n / len(labels)

    sil, dbi, chi = np.nan, np.nan, np.nan
    mask = ~noise_mask
    if n_clusters > 1 and mask.sum() > n_clusters:
        X_clean = X[mask]
        l_clean = labels[mask]
        try:
            sil = silhouette_score(X_clean, l_clean)
        except Exception:
            pass
        try:
            dbi = davies_bouldin_score(X_clean, l_clean)
        except Exception:
            pass
        try:
            chi = calinski_harabasz_score(X_clean, l_clean)
        except Exception:
            pass

    return {
        "method":            tag,
        "n_clusters":        n_clusters,
        "silhouette":        sil,
        "davies_bouldin":    dbi,
        "calinski_harabasz": chi,
        "noise_ratio":       noise_ratio,
        "n_noise":           noise_n,
    }


# =============================================================================
# 1. CARGA
# =============================================================================
print("=" * 65)
print("COMPONENTE 2 — CLUSTERING DE CLIENTES (COBRANZAS)")
print("=" * 65)

if not INPUT_CSV.exists():
    sys.exit(
        f"\n[ERROR FATAL] No se encontro:\n  {INPUT_CSV}\n"
        "Sube features_ml_prepared.csv a la carpeta data/ del script.\n"
    )

df = pd.read_csv(INPUT_CSV)
print(f"[OK] features_ml_prepared.csv cargado: {df.shape}")

if CLIENT_COL not in df.columns:
    sys.exit(
        f"\n[ERROR FATAL] Columna '{CLIENT_COL}' no encontrada.\n"
        f"Columnas disponibles: {list(df.columns)}\n"
    )

# =============================================================================
# 2. RECONSTRUIR sector_dominante PARA PERFILES (solo interpretacion)
# =============================================================================
sector_oh_cols = [
    c for c in df.columns
    if c.startswith(SECTOR_PREFIX)
    and c not in EXCLUDE_ALWAYS
    and c != TARGET_COL
]

if "sector_dominante" not in df.columns and sector_oh_cols:
    df["sector_dominante"] = (
        df[sector_oh_cols]
        .idxmax(axis=1)
        .str.replace(SECTOR_PREFIX, "", regex=False)
    )
    print(f"[OK] 'sector_dominante' reconstruida para perfiles "
          f"(no usada como predictor ordinal).")

# =============================================================================
# 3. SELECCIONAR VARIABLES DE COMPORTAMIENTO DISPONIBLES
# =============================================================================
present_behavior = [c for c in BEHAVIOR_VARS if c in df.columns]
present_cont     = [c for c in CONT_VARS    if c in df.columns]
present_ratio    = [c for c in RATIO_VARS   if c in df.columns]
present_count    = [c for c in COUNT_VARS   if c in df.columns]
present_binary   = [c for c in BINARY_VARS  if c in df.columns]

print(f"\n[OK] Variables de comportamiento disponibles: {len(present_behavior)}")
print(f"  continuas: {len(present_cont)} | ratios: {len(present_ratio)} | "
      f"conteos: {len(present_count)} | binarias: {len(present_binary)}")

if not present_behavior:
    # Fallback: usar todas las numericas disponibles excepto excluidas
    print("[WARN] Ninguna variable prioritaria encontrada. "
          "Usando todas las numericas no excluidas.")
    present_behavior = [
        c for c in df.columns
        if c not in EXCLUDE_ALWAYS
        and "sector_" not in c          # one-hot de sector no al clustering
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    present_cont  = present_behavior
    present_ratio = []
    present_count = []
    present_binary= []

# =============================================================================
# 4. GUARDAR COPIA ORIGINAL ANTES DE LOG1P (para perfiles interpretables)
# =============================================================================
df_orig = df.copy()   # sin log1p, sin escalado — solo para cluster_profiles.csv

log1p_apply = [c for c in LOG1P_VARS if c in df.columns]
if log1p_apply:
    df = safe_log1p(df, log1p_apply)
    print(f"[OK] log1p aplicada sobre {len(log1p_apply)} columnas para clustering.")

# =============================================================================
# 5. AGREGACION CURADA A NIVEL CLIENTE
# =============================================================================
print("\n[STEP] Construyendo tabla agregada por cliente ...")

agg_dict = {}

for c in present_cont:
    agg_dict[c] = ["mean", "max"]

for c in present_ratio:
    if c not in agg_dict:
        agg_dict[c] = ["mean"]

for c in present_count:
    if c not in agg_dict:
        agg_dict[c] = ["sum", "mean"]

for c in present_binary:
    if c not in agg_dict:
        agg_dict[c] = ["mean"]   # tasa de ocurrencia por cliente

# Numero de facturas / registros por cliente
if ID_COL in df.columns:
    n_facturas_series = df.groupby(CLIENT_COL)[ID_COL].nunique().rename("n_facturas_total")
else:
    n_facturas_series = df.groupby(CLIENT_COL).size().rename("n_facturas_total")

client_agg = df.groupby(CLIENT_COL).agg(agg_dict)
client_agg.columns = ["_".join(col).strip() for col in client_agg.columns]
client_agg = client_agg.reset_index()
client_agg = client_agg.merge(
    n_facturas_series.reset_index(), on=CLIENT_COL, how="left"
)

print(f"[OK] Tabla agregada: {client_agg.shape} — "
      f"{client_agg[CLIENT_COL].nunique()} clientes unicos")

# ── Tabla agregada RAW (sin log1p) para perfiles interpretables ─────────────
if ID_COL in df_orig.columns:
    n_facturas_raw = df_orig.groupby(CLIENT_COL)[ID_COL].nunique().rename("n_facturas_total")
else:
    n_facturas_raw = df_orig.groupby(CLIENT_COL).size().rename("n_facturas_total")

client_agg_raw = df_orig.groupby(CLIENT_COL).agg(agg_dict)
client_agg_raw.columns = ["_".join(col).strip() for col in client_agg_raw.columns]
client_agg_raw = client_agg_raw.reset_index()
client_agg_raw = client_agg_raw.merge(
    n_facturas_raw.reset_index(), on=CLIENT_COL, how="left"
)
print(f"[OK] Tabla raw (sin log1p) para perfiles: {client_agg_raw.shape}")

# =============================================================================
# 6. PREPARACION PARA CLUSTERING
# =============================================================================
client_ids = client_agg[CLIENT_COL].values
X_raw = client_agg.drop(columns=[CLIENT_COL]).copy()

# ── Imputacion tipificada (no fillna(0) global) ──────────────────────────────
# Identificar nombres de columnas agregadas por tipo de variable origen
_cont_agg_cols  = [c for c in X_raw.columns
                   if any(c.startswith(v + "_") for v in present_cont + present_ratio)]
_count_agg_cols = [c for c in X_raw.columns
                   if any(c.startswith(v + "_") for v in present_count)]
_bin_agg_cols   = [c for c in X_raw.columns
                   if any(c.startswith(v + "_") for v in present_binary)]
_other_cols     = [c for c in X_raw.columns
                   if c not in _cont_agg_cols + _count_agg_cols + _bin_agg_cols]

# Continuas y ratios → mediana
for c in _cont_agg_cols:
    X_raw[c] = X_raw[c].fillna(X_raw[c].median())

# Conteos → 0 (ausencia equivale a ninguna ocurrencia)
for c in _count_agg_cols:
    X_raw[c] = X_raw[c].fillna(0)

# Binarias (tasa de ocurrencia) → 0
for c in _bin_agg_cols:
    X_raw[c] = X_raw[c].fillna(0)

# Resto (n_facturas_total u otras) → mediana
for c in _other_cols:
    X_raw[c] = X_raw[c].fillna(X_raw[c].median())

print(f"[OK] Imputacion tipificada: {len(_cont_agg_cols)} continuas/ratios (mediana), "
      f"{len(_count_agg_cols)} conteos (0), {len(_bin_agg_cols)} binarias (0), "
      f"{len(_other_cols)} otras (mediana)")

# Eliminar columnas con varianza cero
zero_var_cols = X_raw.columns[X_raw.var() == 0].tolist()
if zero_var_cols:
    print(f"[INFO] Eliminando {len(zero_var_cols)} columnas con varianza cero.")
    X_raw = X_raw.drop(columns=zero_var_cols)

feature_names = X_raw.columns.tolist()
print(f"[OK] Features finales para clustering: {len(feature_names)}")

# Escalar con RobustScaler (resistente a outliers)
scaler  = RobustScaler()
X_scaled = scaler.fit_transform(X_raw.values)

# =============================================================================
# 7. KMEANS — BUSQUEDA AUXILIAR DE K OPTIMO (silhouette)
# =============================================================================
print("\n[STEP] Busqueda auxiliar de k optimo (silhouette) ...")

k_search_rows = []
for k in KMEANS_K_SEARCH:
    if k >= len(client_ids):
        break
    km_tmp = KMeans(n_clusters=k, random_state=KMEANS_SEED, n_init=10)
    lbl_tmp = km_tmp.fit_predict(X_scaled)
    sil_tmp = silhouette_score(X_scaled, lbl_tmp) if len(set(lbl_tmp)) > 1 else -1
    k_search_rows.append({
        "method":            f"KMeans_k{k}_search",
        "k_evaluated":       k,
        "inertia":           km_tmp.inertia_,
        "silhouette":        sil_tmp,
        "davies_bouldin":    np.nan,
        "calinski_harabasz": np.nan,
        "noise_ratio":       np.nan,
        "n_noise":           np.nan,
        "n_clusters":        k,
    })
    print(f"  k={k:2d} | inertia={km_tmp.inertia_:.2f} | silhouette={sil_tmp:.4f}")

best_k_sil = max(k_search_rows, key=lambda r: r["silhouette"])["k_evaluated"] \
             if k_search_rows else KMEANS_K_FIXED
print(f"[INFO] k optimo por silhouette: {best_k_sil} "
      f"| k principal del proyecto: {KMEANS_K_FIXED}")

# =============================================================================
# 8. KMEANS PRINCIPAL (k fijo alineado al proyecto)
# =============================================================================
print(f"\n[STEP] KMeans principal con k={KMEANS_K_FIXED} ...")
kmeans_main = KMeans(n_clusters=KMEANS_K_FIXED, random_state=KMEANS_SEED, n_init=20)
km_labels   = kmeans_main.fit_predict(X_scaled)
km_metrics  = cluster_metrics(X_scaled, km_labels, f"KMeans_k{KMEANS_K_FIXED}")

print(f"  Silhouette:        {km_metrics['silhouette']:.4f}")
print(f"  Davies-Bouldin:    {km_metrics['davies_bouldin']:.4f}")
print(f"  Calinski-Harabasz: {km_metrics['calinski_harabasz']:.4f}")

# KMeans con k optimo por silhouette (si difiere del fijo)
km_labels_opt = None
km_metrics_opt = {}
if best_k_sil != KMEANS_K_FIXED:
    print(f"\n[STEP] KMeans auxiliar con k={best_k_sil} (k optimo silhouette) ...")
    kmeans_opt    = KMeans(n_clusters=best_k_sil, random_state=KMEANS_SEED, n_init=20)
    km_labels_opt = kmeans_opt.fit_predict(X_scaled)
    km_metrics_opt = cluster_metrics(X_scaled, km_labels_opt, f"KMeans_k{best_k_sil}_optimal")
    print(f"  Silhouette:        {km_metrics_opt['silhouette']:.4f}")

# =============================================================================
# 9. DBSCAN COMPARATIVO
# =============================================================================
effective_min = min(DBSCAN_MIN_SAMPLES, max(2, len(client_ids) // 20))
print(f"\n[STEP] DBSCAN (eps={DBSCAN_EPS}, min_samples={effective_min}) ...")
dbscan     = DBSCAN(eps=DBSCAN_EPS, min_samples=effective_min)
db_labels  = dbscan.fit_predict(X_scaled)
db_metrics = cluster_metrics(X_scaled, db_labels, "DBSCAN")

print(f"  Clusters:          {db_metrics['n_clusters']}")
print(f"  Ruido:             {db_metrics['n_noise']} ({db_metrics['noise_ratio']:.2%})")
if not np.isnan(db_metrics["silhouette"]):
    print(f"  Silhouette:        {db_metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin:    {db_metrics['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz: {db_metrics['calinski_harabasz']:.4f}")

# =============================================================================
# 10. EXPORTAR clustering_metrics.csv
# =============================================================================
metrics_rows = [km_metrics, db_metrics]
if km_metrics_opt:
    metrics_rows.append(km_metrics_opt)
metrics_rows += k_search_rows

clustering_metrics_df = pd.DataFrame(metrics_rows)
# Columna k_evaluated puede no existir en filas principales — rellenar
if "k_evaluated" not in clustering_metrics_df.columns:
    clustering_metrics_df["k_evaluated"] = np.nan
else:
    clustering_metrics_df["k_evaluated"] = clustering_metrics_df["k_evaluated"].fillna(
        clustering_metrics_df["n_clusters"]
    )

metrics_path = DATA_DIR / "clustering_metrics.csv"
clustering_metrics_df.to_csv(metrics_path, index=False)
print(f"\n[SAVED] {metrics_path}")

# =============================================================================
# 11. EXPORTAR client_cluster_assignments.csv
# =============================================================================
assignments_df = pd.DataFrame({
    CLIENT_COL:                   client_ids,
    f"cluster_kmeans_k{KMEANS_K_FIXED}": km_labels,
    "cluster_dbscan":              db_labels,
})
if km_labels_opt is not None:
    assignments_df[f"cluster_kmeans_k{best_k_sil}_optimal"] = km_labels_opt

assignments_path = DATA_DIR / "client_cluster_assignments.csv"
assignments_df.to_csv(assignments_path, index=False)
print(f"[SAVED] {assignments_path}")

# =============================================================================
# 12. EXPORTAR cluster_profiles.csv (KMeans principal)
# =============================================================================
print("\n[STEP] Calculando perfiles por cluster ...")

# Perfiles construidos desde datos RAW (sin log1p, sin escalado)
# para que los valores sean interpretables en escala de negocio original
X_raw_profile = client_agg_raw.drop(columns=[CLIENT_COL]).copy()

# Imputar X_raw_profile con la misma logica tipificada
for c in [col for col in X_raw_profile.columns if col in _cont_agg_cols]:
    X_raw_profile[c] = X_raw_profile[c].fillna(X_raw_profile[c].median())
for c in [col for col in X_raw_profile.columns if col in _count_agg_cols + _bin_agg_cols]:
    X_raw_profile[c] = X_raw_profile[c].fillna(0)
for c in [col for col in X_raw_profile.columns
          if col not in _cont_agg_cols + _count_agg_cols + _bin_agg_cols]:
    X_raw_profile[c] = X_raw_profile[c].fillna(X_raw_profile[c].median())

# Alinear columnas al mismo orden y conjunto que X_raw (pueden diferir por zero_var)
profile_feature_names = [c for c in feature_names if c in X_raw_profile.columns]
X_raw_profile = X_raw_profile[profile_feature_names]

profile_base = X_raw_profile.copy()
profile_base[CLIENT_COL]       = client_ids
profile_base["cluster_kmeans"] = km_labels

# Media de variables agregadas por cluster (en escala original, interpretable)
profile_df = (
    profile_base
    .groupby("cluster_kmeans")[profile_feature_names]
    .mean()
    .reset_index()
    .rename(columns={"cluster_kmeans": "cluster"})
)

# Tamano de cada cluster
sizes = (
    pd.Series(km_labels)
    .value_counts()
    .sort_index()
    .reset_index()
)
sizes.columns = ["cluster", "n_clientes"]
profile_df = profile_df.merge(sizes, on="cluster", how="left")

# Sector dominante modal (sin ordinalidad artificial)
if "sector_dominante" in df.columns:
    # Unir sector_dominante original a nivel cliente (moda)
    sector_modal = (
        df.groupby(CLIENT_COL)["sector_dominante"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .reset_index()
    )
    sector_modal.columns = [CLIENT_COL, "sector_dominante_modal"]
    tmp = assignments_df[[CLIENT_COL, f"cluster_kmeans_k{KMEANS_K_FIXED}"]].copy()
    tmp.columns = [CLIENT_COL, "cluster_kmeans"]
    tmp = tmp.merge(sector_modal, on=CLIENT_COL, how="left")
    sector_by_cluster = (
        tmp.groupby("cluster_kmeans")["sector_dominante_modal"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A")
        .reset_index()
        .rename(columns={
            "cluster_kmeans": "cluster",
            "sector_dominante_modal": "sector_dominante_modal"
        })
    )
    profile_df = profile_df.merge(sector_by_cluster, on="cluster", how="left")
    print("[OK] Sector dominante modal agregado a perfiles.")

# Reordenar columnas: cluster, n_clientes al frente
front_cols = ["cluster", "n_clientes"]
if "sector_dominante_modal" in profile_df.columns:
    front_cols.append("sector_dominante_modal")
other_cols = [c for c in profile_df.columns if c not in front_cols]
profile_df = profile_df[front_cols + other_cols]

profiles_path = DATA_DIR / "cluster_profiles.csv"
profile_df.to_csv(profiles_path, index=False)
print(f"[SAVED] {profiles_path}")

# =============================================================================
# 13. RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 65)
print("RESUMEN — COMPONENTE 2 CLUSTERING DE CLIENTES")
print("=" * 65)
print(f"  Clientes analizados:       {len(client_ids):,}")
print(f"  Features de comportamiento: {len(feature_names)}")
print(f"  Log1p aplicado a:          {len(log1p_apply)} columnas")
print(f"  Escalado:                  RobustScaler")
print("")
print(f"  KMeans k={KMEANS_K_FIXED} (principal)")
print(f"    Silhouette:              {km_metrics['silhouette']:.4f}")
print(f"    Davies-Bouldin:          {km_metrics['davies_bouldin']:.4f}")
print(f"    Calinski-Harabasz:       {km_metrics['calinski_harabasz']:.4f}")
print("")
print(f"  KMeans k={best_k_sil} (optimo silhouette)")
if km_metrics_opt:
    print(f"    Silhouette:              {km_metrics_opt['silhouette']:.4f}")
else:
    print(f"    (coincide con k principal)")
print("")
print(f"  DBSCAN")
print(f"    Clusters:                {db_metrics['n_clusters']}")
print(f"    Noise ratio:             {db_metrics['noise_ratio']:.2%}")
sil_db = db_metrics['silhouette']
print(f"    Silhouette (sin ruido):  {sil_db:.4f}" if not np.isnan(sil_db) else
      f"    Silhouette:              N/A (clusters insuficientes)")

print("\n  Distribucion de clientes por cluster (KMeans principal):")
for _, row in profile_df[["cluster", "n_clientes"]].iterrows():
    pct = row["n_clientes"] / len(client_ids) * 100
    bar = "#" * int(pct / 2)
    print(f"    Cluster {int(row['cluster'])}: {int(row['n_clientes']):>5} clientes "
          f"({pct:5.1f}%) {bar}")

print("\n  Archivos exportados:")
for fname in ["clustering_metrics.csv", "client_cluster_assignments.csv", "cluster_profiles.csv"]:
    p = DATA_DIR / fname
    status = "OK" if p.exists() else "FALTANTE"
    print(f"    [{status}] {p}")

print("=" * 65)
print("[OK] Componente 2 completado exitosamente.")
