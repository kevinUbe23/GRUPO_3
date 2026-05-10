"""Clustering de clientes para segmentacion de cobranza."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PREP_OUTPUTS = PROJECT_DIR / "03_preparacion" / "outputs"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIENT_FEATURES_CSV = PREP_OUTPUTS / "client_features_clustering_base.csv"
CLIENT_FEATURE_LIST_CSV = PREP_OUTPUTS / "client_clustering_features_selected.csv"

CLIENT_COL = "cliente_id"
SECTOR_PROFILE_COL = "sector_dominante_modal"
KMEANS_K_FIXED = 3
KMEANS_K_SEARCH = range(2, 6)
KMEANS_SEED = 42
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
DBSCAN_EPS_SEARCH = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
TOP_DRIVER_FEATURES = 8
TOP_CLIENT_REASON_FEATURES = 6

RATING_RISK_WEIGHTS = {
    "mora_promedio_hist_mean": 1.4,
    "mora_ultimo_tramo_mean": 1.4,
    "dias_mora_observable_mean": 1.3,
    "moras_consecutivas_mean": 1.3,
    "ratio_promesas_rotas_mean": 1.2,
    "friccion_contacto_mean": 1.1,
    "num_no_contesta_cons_mean": 0.9,
    "tiene_disputa_activa_mean": 0.8,
    "esta_vencida_al_corte_mean": 0.8,
}

RATING_PROTECTIVE_WEIGHTS = {
    "tasa_cumplimiento_mean": 1.4,
    "tasa_contacto_cliente_mean": 1.0,
    "tasa_cumpl_promesas_mean": 1.2,
    "dias_hasta_vence_pos_mean": 0.6,
}

RATING_REASON_LABELS = {
    "mora_promedio_hist_mean": "mora historica",
    "mora_ultimo_tramo_mean": "mora reciente",
    "dias_mora_observable_mean": "dias de mora observables",
    "moras_consecutivas_mean": "moras consecutivas",
    "ratio_promesas_rotas_mean": "promesas rotas",
    "friccion_contacto_mean": "friccion de contacto",
    "num_no_contesta_cons_mean": "no contestacion",
    "tiene_disputa_activa_mean": "disputas activas",
    "esta_vencida_al_corte_mean": "facturas vencidas al corte",
    "tasa_cumplimiento_mean": "bajo cumplimiento historico",
    "tasa_contacto_cliente_mean": "baja contactabilidad",
    "tasa_cumpl_promesas_mean": "bajo cumplimiento de promesas",
    "dias_hasta_vence_pos_mean": "poco margen preventivo antes de vencer",
}

LOG1P_BASE_VARS = [
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

KEY_PROFILE_FEATURES = [
    "monto_mean",
    "mora_promedio_hist_mean",
    "mora_ultimo_tramo_mean",
    "dias_mora_observable_mean",
    "dias_hasta_vence_pos_mean",
    "friccion_contacto_mean",
    "tasa_contacto_cliente_mean",
    "tasa_cumplimiento_mean",
    "tasa_cumpl_promesas_mean",
    "ratio_promesas_rotas_mean",
    "num_gestiones_factura_mean",
    "num_promesas_rotas_mean",
    "promesas_total_mean",
    "tiene_disputa_activa_mean",
    "esta_vencida_al_corte_mean",
    "n_facturas_total",
    "n_cortes_total",
]


def needs_log1p(column: str) -> bool:
    return any(column.startswith(f"{base}_") for base in LOG1P_BASE_VARS)


def prepare_clustering_matrix(client_features: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    X_raw = client_features[feature_cols].copy()
    log1p_cols = [col for col in X_raw.columns if needs_log1p(col)]

    for col in X_raw.columns:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
        if col in log1p_cols:
            X_raw[col] = np.log1p(np.maximum(X_raw[col].fillna(0), 0))
        elif X_raw[col].isna().any():
            X_raw[col] = X_raw[col].fillna(X_raw[col].median())

    zero_var_cols = X_raw.columns[X_raw.var() == 0].tolist()
    if zero_var_cols:
        X_raw = X_raw.drop(columns=zero_var_cols)

    return X_raw, X_raw.columns.tolist(), log1p_cols


def cluster_metrics(X: np.ndarray, labels: np.ndarray, tag: str) -> dict:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_mask = labels == -1
    noise_n = int(noise_mask.sum())
    noise_ratio = noise_n / len(labels)

    sil, dbi, chi = np.nan, np.nan, np.nan
    mask = ~noise_mask
    if n_clusters > 1 and mask.sum() > n_clusters:
        X_clean = X[mask]
        labels_clean = labels[mask]
        try:
            sil = silhouette_score(X_clean, labels_clean)
        except Exception:
            pass
        try:
            dbi = davies_bouldin_score(X_clean, labels_clean)
        except Exception:
            pass
        try:
            chi = calinski_harabasz_score(X_clean, labels_clean)
        except Exception:
            pass

    return {
        "method": tag,
        "n_clusters": n_clusters,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": chi,
        "noise_ratio": noise_ratio,
        "n_noise": noise_n,
    }


def feature_label(feature: str) -> str:
    return (
        feature
        .replace("_mean", " prom.")
        .replace("_max", " max.")
        .replace("_sum", " total")
        .replace("_", " ")
    )


def save_k_search_plot(k_search_rows: list[dict]) -> None:
    if not k_search_rows:
        return

    df = pd.DataFrame(k_search_rows).sort_values("k_evaluated")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df["k_evaluated"], df["silhouette"], marker="o", color="#1f77b4")
    ax1.set_xlabel("Numero de clusters k")
    ax1.set_ylabel("Silhouette", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(df["k_evaluated"])

    ax2 = ax1.twinx()
    ax2.plot(df["k_evaluated"], df["inertia"], marker="s", color="#d62728")
    ax2.set_ylabel("Inertia", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("Busqueda de k para KMeans")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "clustering_k_search.png", dpi=160)
    plt.close(fig)


def save_cluster_size_plot(profile_df: pd.DataFrame, total_clients: int) -> None:
    plot_df = profile_df[["cluster", "n_clientes"]].copy()
    plot_df["porcentaje"] = plot_df["n_clientes"] / total_clients * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=plot_df, x="cluster", y="n_clientes", color="#4c78a8", ax=ax)
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            f"{int(row['n_clientes'])}\n{row['porcentaje']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Clientes por cluster - KMeans principal")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Clientes")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_sizes.png", dpi=160)
    plt.close(fig)


def save_pca_plot(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    client_ids: np.ndarray,
    client_features: pd.DataFrame,
) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=KMEANS_SEED)
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        CLIENT_COL: client_ids,
        "cluster": labels,
        "pca_1": coords[:, 0],
        "pca_2": coords[:, 1],
        "pca_1_variance_ratio": pca.explained_variance_ratio_[0],
        "pca_2_variance_ratio": pca.explained_variance_ratio_[1],
    })
    if SECTOR_PROFILE_COL in client_features.columns:
        pca_df[SECTOR_PROFILE_COL] = client_features[SECTOR_PROFILE_COL].astype(str).values

    pca_df.to_csv(OUTPUT_DIR / "cluster_pca_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="pca_1",
        y="pca_2",
        hue="cluster",
        palette="tab10",
        s=70,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title("Mapa 2D de clientes por cluster (PCA)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_pca_scatter.png", dpi=160)
    plt.close(fig)

    return pca_df


def build_feature_drivers(
    profile_base: pd.DataFrame,
    feature_names: list[str],
    cluster_centers: np.ndarray,
) -> pd.DataFrame:
    global_means = profile_base[feature_names].mean()
    global_medians = profile_base[feature_names].median()
    cluster_means = profile_base.groupby("cluster_kmeans")[feature_names].mean()
    cluster_medians = profile_base.groupby("cluster_kmeans")[feature_names].median()

    rows = []
    for cluster_id in sorted(profile_base["cluster_kmeans"].unique()):
        for idx, feature in enumerate(feature_names):
            global_mean = global_means[feature]
            cluster_mean = cluster_means.loc[cluster_id, feature]
            if pd.isna(global_mean) or np.isclose(global_mean, 0):
                ratio = np.nan
                pct_diff = np.nan
            else:
                ratio = cluster_mean / global_mean
                pct_diff = (ratio - 1) * 100

            centroid_value = cluster_centers[int(cluster_id), idx]
            rows.append({
                "cluster": int(cluster_id),
                "feature": feature,
                "feature_label": feature_label(feature),
                "cluster_mean": cluster_mean,
                "global_mean": global_mean,
                "cluster_median": cluster_medians.loc[cluster_id, feature],
                "global_median": global_medians[feature],
                "cluster_to_global_mean_ratio": ratio,
                "pct_diff_vs_global_mean": pct_diff,
                "centroid_scaled_value": centroid_value,
                "abs_centroid_scaled_value": abs(centroid_value),
                "direction": "alto" if centroid_value > 0 else "bajo",
            })

    drivers_df = pd.DataFrame(rows)
    drivers_df["rank_in_cluster"] = (
        drivers_df
        .groupby("cluster")["abs_centroid_scaled_value"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    drivers_df = drivers_df.sort_values(["cluster", "rank_in_cluster"])
    return drivers_df


def save_driver_heatmap(drivers_df: pd.DataFrame) -> None:
    top_features = (
        drivers_df
        .groupby("feature", as_index=False)["abs_centroid_scaled_value"]
        .max()
        .sort_values("abs_centroid_scaled_value", ascending=False)
        .head(18)["feature"]
        .tolist()
    )
    if not top_features:
        return

    heatmap_df = (
        drivers_df[drivers_df["feature"].isin(top_features)]
        .pivot(index="cluster", columns="feature_label", values="centroid_scaled_value")
    )

    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.heatmap(
        heatmap_df,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Variables que mas diferencian a cada cluster")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Cluster")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_feature_drivers_heatmap.png", dpi=160)
    plt.close(fig)


def save_key_profile_heatmap(profile_df: pd.DataFrame, feature_names: list[str]) -> None:
    selected_features = [f for f in KEY_PROFILE_FEATURES if f in feature_names]
    if not selected_features:
        return

    profile_matrix = profile_df.set_index("cluster")[selected_features].copy()
    global_mean = profile_matrix.mean()
    global_std = profile_matrix.std(ddof=0).replace(0, np.nan)
    standardized = (profile_matrix - global_mean) / global_std
    standardized = standardized.fillna(0)
    standardized.columns = [feature_label(col) for col in standardized.columns]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.heatmap(
        standardized,
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Perfil comparativo de clusters en variables clave")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Cluster")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_key_profile_heatmap.png", dpi=160)
    plt.close(fig)


def build_client_distance_tables(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    client_ids: np.ndarray,
    cluster_centers: np.ndarray,
    client_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    distances = pairwise_distances(X_scaled, cluster_centers, metric="euclidean")
    distance_cols = [f"distance_to_cluster_{i}" for i in range(cluster_centers.shape[0])]
    distances_df = pd.DataFrame(distances, columns=distance_cols)
    distances_df.insert(0, CLIENT_COL, client_ids)
    distances_df.insert(1, "cluster", labels)

    sorted_distances = np.sort(distances, axis=1)
    distances_df["assigned_distance"] = sorted_distances[:, 0]
    distances_df["second_nearest_distance"] = sorted_distances[:, 1]
    distances_df["margin_to_second_nearest"] = (
        distances_df["second_nearest_distance"] - distances_df["assigned_distance"]
    )
    distances_df["relative_margin"] = (
        distances_df["margin_to_second_nearest"]
        / distances_df["second_nearest_distance"].replace(0, np.nan)
    )

    context_cols = [
        col for col in [
            SECTOR_PROFILE_COL,
            "n_facturas_total",
            "n_cortes_total",
            "monto_mean",
            "mora_promedio_hist_mean",
            "dias_mora_observable_mean",
            "friccion_contacto_mean",
            "tasa_cumplimiento_mean",
            "ratio_promesas_rotas_mean",
        ]
        if col in client_features.columns
    ]
    detailed_df = pd.concat(
        [
            distances_df,
            client_features[context_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    example_rows = []
    for cluster_id, cluster_df in detailed_df.groupby("cluster"):
        rep = cluster_df.sort_values("assigned_distance").head(3).copy()
        rep["example_type"] = "representativo"

        boundary = (
            cluster_df[~cluster_df[CLIENT_COL].isin(rep[CLIENT_COL])]
            .sort_values("margin_to_second_nearest")
            .head(3)
            .copy()
        )
        boundary["example_type"] = "frontera"

        example_rows.append(rep)
        if not boundary.empty:
            example_rows.append(boundary)

    examples_df = pd.concat(example_rows, ignore_index=True)
    return detailed_df, examples_df


def build_client_reasoning(
    client_features: pd.DataFrame,
    assignments_detailed_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
) -> pd.DataFrame:
    top_driver_map = {
        cluster_id: group.head(TOP_CLIENT_REASON_FEATURES)
        for cluster_id, group in drivers_df.groupby("cluster", sort=True)
    }

    client_lookup = client_features.set_index(CLIENT_COL)
    rows = []
    for _, assignment in assignments_detailed_df.iterrows():
        client_id = assignment[CLIENT_COL]
        cluster_id = int(assignment["cluster"])
        if cluster_id not in top_driver_map or client_id not in client_lookup.index:
            continue

        client_row = client_lookup.loc[client_id]
        for _, driver in top_driver_map[cluster_id].iterrows():
            feature = driver["feature"]
            client_value = client_row[feature]
            global_mean = driver["global_mean"]
            if pd.isna(global_mean) or np.isclose(global_mean, 0):
                client_vs_global_ratio = np.nan
            else:
                client_vs_global_ratio = client_value / global_mean

            rows.append({
                CLIENT_COL: client_id,
                "cluster": cluster_id,
                "feature": feature,
                "feature_label": driver["feature_label"],
                "cluster_driver_rank": driver["rank_in_cluster"],
                "cluster_direction": driver["direction"],
                "client_value": client_value,
                "cluster_mean": driver["cluster_mean"],
                "global_mean": global_mean,
                "client_to_global_mean_ratio": client_vs_global_ratio,
                "assigned_distance": assignment["assigned_distance"],
                "margin_to_second_nearest": assignment["margin_to_second_nearest"],
            })

    return pd.DataFrame(rows)


def build_cluster_readable_summary(
    profile_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    total_clients: int,
) -> pd.DataFrame:
    rows = []
    for cluster_id, group in drivers_df.groupby("cluster", sort=True):
        high_features = (
            group[group["direction"] == "alto"]
            .head(5)["feature_label"]
            .tolist()
        )
        low_features = (
            group[group["direction"] == "bajo"]
            .head(5)["feature_label"]
            .tolist()
        )
        profile_row = profile_df[profile_df["cluster"] == cluster_id].iloc[0]

        rows.append({
            "cluster": int(cluster_id),
            "n_clientes": int(profile_row["n_clientes"]),
            "porcentaje_clientes": profile_row["n_clientes"] / total_clients,
            "sector_dominante_modal": profile_row.get(SECTOR_PROFILE_COL, "N/A"),
            "variables_altas": " | ".join(high_features),
            "variables_bajas": " | ".join(low_features),
        })

    return pd.DataFrame(rows)


def percentile_01(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True).fillna(0.5)


def risk_score_to_stars(score: float) -> int:
    if score <= 20:
        return 5
    if score <= 40:
        return 4
    if score <= 60:
        return 3
    if score <= 80:
        return 2
    return 1


def star_label(stars: int) -> str:
    return f"{stars} estrella" if stars == 1 else f"{stars} estrellas"


def segment_label_for_cluster(row: pd.Series, max_risk_cluster: int, min_risk_cluster: int) -> str:
    cluster_id = int(row["cluster"])
    n_clientes = int(row["n_clientes"])
    if n_clientes <= 5:
        return "Clientes atipicos de bajo volumen"
    if cluster_id == max_risk_cluster:
        return "Clientes de alto riesgo operativo"
    if cluster_id == min_risk_cluster:
        return "Clientes preventivos o de menor riesgo"
    return "Clientes recurrentes de comportamiento general"


def build_frontend_customer_view(
    client_rating_df: pd.DataFrame,
    cluster_rating_df: pd.DataFrame,
    client_reasoning_df: pd.DataFrame,
    rating_reasoning_df: pd.DataFrame,
) -> pd.DataFrame:
    cluster_lookup = cluster_rating_df.set_index("cluster")["tipo_cliente"].to_dict()
    cluster_reason_map = (
        client_reasoning_df
        .sort_values([CLIENT_COL, "cluster_driver_rank"])
        .groupby(CLIENT_COL)["feature_label"]
        .apply(lambda values: " | ".join(values.head(4)))
        .to_dict()
    )
    rating_reason_map = (
        rating_reasoning_df
        .sort_values([CLIENT_COL, "rating_reason_rank"])
        .groupby(CLIENT_COL)["reason_label"]
        .apply(lambda values: " | ".join(values.head(4)))
        .to_dict()
    )

    view_cols = [
        CLIENT_COL,
        "cluster",
        "client_risk_score_0_100",
        "rating_estrellas",
        "rating_label",
        SECTOR_PROFILE_COL,
        "n_facturas_total",
        "n_cortes_total",
    ]
    view_cols = [col for col in view_cols if col in client_rating_df.columns]
    frontend_df = client_rating_df[view_cols].copy()
    frontend_df["tipo_cliente"] = frontend_df["cluster"].map(cluster_lookup)
    frontend_df["por_que_rating"] = frontend_df[CLIENT_COL].map(rating_reason_map)
    frontend_df["por_que_cluster"] = (
        "Cercania al centroide del grupo; variables distintivas: "
        + frontend_df[CLIENT_COL].map(cluster_reason_map).fillna("N/A")
    )
    frontend_df = frontend_df.rename(columns={
        CLIENT_COL: "cliente_id",
        "client_risk_score_0_100": "riesgo_0_100",
    })
    front_order = [
        "cliente_id",
        "tipo_cliente",
        "cluster",
        "riesgo_0_100",
        "rating_estrellas",
        "rating_label",
        "por_que_rating",
        "por_que_cluster",
        SECTOR_PROFILE_COL,
        "n_facturas_total",
        "n_cortes_total",
    ]
    front_order = [col for col in front_order if col in frontend_df.columns]
    return frontend_df[front_order].sort_values(["rating_estrellas", "riesgo_0_100", "cliente_id"])


def build_cluster_ratings(
    client_features: pd.DataFrame,
    assignments_detailed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rating_base = pd.DataFrame({
        CLIENT_COL: client_features[CLIENT_COL].astype(str),
    })

    component_rows = []
    component_values = []
    total_weight = 0.0

    for feature, weight in RATING_RISK_WEIGHTS.items():
        if feature not in client_features.columns:
            continue
        normalized = percentile_01(pd.to_numeric(client_features[feature], errors="coerce"))
        component_values.append((feature, weight, normalized, "mayor_valor_mayor_riesgo"))
        total_weight += weight
        component_rows.append({
            "feature": feature,
            "feature_label": feature_label(feature),
            "reason_label": RATING_REASON_LABELS.get(feature, feature_label(feature)),
            "weight": weight,
            "direction": "mayor_valor_mayor_riesgo",
            "interpretation": "Aumenta riesgo de cobranza y reduce estrellas.",
        })

    for feature, weight in RATING_PROTECTIVE_WEIGHTS.items():
        if feature not in client_features.columns:
            continue
        normalized = 1 - percentile_01(pd.to_numeric(client_features[feature], errors="coerce"))
        component_values.append((feature, weight, normalized, "mayor_valor_menor_riesgo"))
        total_weight += weight
        component_rows.append({
            "feature": feature,
            "feature_label": feature_label(feature),
            "reason_label": RATING_REASON_LABELS.get(feature, feature_label(feature)),
            "weight": weight,
            "direction": "mayor_valor_menor_riesgo",
            "interpretation": "Mejor comportamiento; se invierte para el score de riesgo.",
        })

    if not component_values or total_weight == 0:
        raise ValueError("No hay variables disponibles para calcular rating de estrellas.")

    weighted_components = [
        normalized * weight
        for _, weight, normalized, _ in component_values
    ]
    rating_base["client_risk_score_0_100"] = (
        sum(weighted_components) / total_weight * 100
    ).round(2)
    rating_base["rating_estrellas"] = rating_base["client_risk_score_0_100"].map(risk_score_to_stars)
    rating_base["rating_label"] = rating_base["rating_estrellas"].map(star_label)

    rating_base = rating_base.merge(
        assignments_detailed_df[
            [
                CLIENT_COL,
                "cluster",
                "assigned_distance",
                "margin_to_second_nearest",
                "relative_margin",
            ]
        ],
        on=CLIENT_COL,
        how="left",
    )

    context_cols = [
        col for col in [
            SECTOR_PROFILE_COL,
            "n_facturas_total",
            "n_cortes_total",
            "mora_promedio_hist_mean",
            "mora_ultimo_tramo_mean",
            "dias_mora_observable_mean",
            "moras_consecutivas_mean",
            "ratio_promesas_rotas_mean",
            "friccion_contacto_mean",
            "tasa_cumplimiento_mean",
            "tasa_contacto_cliente_mean",
            "tasa_cumpl_promesas_mean",
            "tiene_disputa_activa_mean",
            "esta_vencida_al_corte_mean",
        ]
        if col in client_features.columns
    ]
    rating_base = rating_base.merge(
        client_features[[CLIENT_COL, *context_cols]],
        on=CLIENT_COL,
        how="left",
    )

    cluster_rating_df = (
        rating_base
        .groupby("cluster", as_index=False)
        .agg(
            n_clientes=(CLIENT_COL, "count"),
            cluster_risk_score_0_100=("client_risk_score_0_100", "mean"),
            min_client_risk_score_0_100=("client_risk_score_0_100", "min"),
            max_client_risk_score_0_100=("client_risk_score_0_100", "max"),
            rating_promedio=("rating_estrellas", "mean"),
        )
    )
    cluster_rating_df["cluster_risk_rank"] = (
        cluster_rating_df["cluster_risk_score_0_100"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    cluster_rating_df["rating_cluster_promedio"] = (
        cluster_rating_df["cluster_risk_score_0_100"]
        .map(risk_score_to_stars)
    )
    cluster_rating_df["rating_cluster_label"] = cluster_rating_df["rating_cluster_promedio"].map(star_label)

    rating_counts = (
        rating_base
        .pivot_table(
            index="cluster",
            columns="rating_estrellas",
            values=CLIENT_COL,
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    rating_counts.columns = [
        "cluster" if col == "cluster" else f"clientes_{int(col)}_estrellas"
        for col in rating_counts.columns
    ]
    cluster_rating_df = cluster_rating_df.merge(rating_counts, on="cluster", how="left")

    max_risk_cluster = int(
        cluster_rating_df.sort_values("cluster_risk_score_0_100", ascending=False).iloc[0]["cluster"]
    )
    min_risk_cluster = int(
        cluster_rating_df.sort_values("cluster_risk_score_0_100", ascending=True).iloc[0]["cluster"]
    )
    cluster_rating_df["tipo_cliente"] = cluster_rating_df.apply(
        lambda row: segment_label_for_cluster(row, max_risk_cluster, min_risk_cluster),
        axis=1,
    )
    cluster_rating_df = cluster_rating_df.sort_values("cluster_risk_score_0_100", ascending=False)

    rating_base = rating_base.merge(
        cluster_rating_df[
            [
                "cluster",
                "cluster_risk_score_0_100",
                "cluster_risk_rank",
                "rating_cluster_promedio",
                "rating_cluster_label",
                "tipo_cliente",
            ]
        ],
        on="cluster",
        how="left",
    )
    rating_base = rating_base.sort_values(
        ["rating_estrellas", "client_risk_score_0_100", CLIENT_COL],
        ascending=[True, False, True],
    )

    weights_df = pd.DataFrame(component_rows)

    reasoning_rows = []
    for feature, weight, normalized, direction in component_values:
        contribution_points = (normalized * weight / total_weight * 100).round(2)
        feature_values = pd.to_numeric(client_features[feature], errors="coerce")
        for idx, client_id in enumerate(client_features[CLIENT_COL].astype(str)):
            reasoning_rows.append({
                CLIENT_COL: client_id,
                "feature": feature,
                "feature_label": feature_label(feature),
                "reason_label": RATING_REASON_LABELS.get(feature, feature_label(feature)),
                "direction": direction,
                "weight": weight,
                "client_value": feature_values.iloc[idx],
                "normalized_risk_component": round(float(normalized.iloc[idx]), 4),
                "risk_contribution_points": float(contribution_points.iloc[idx]),
            })

    rating_reasoning_df = pd.DataFrame(reasoning_rows)
    rating_reasoning_df["rating_reason_rank"] = (
        rating_reasoning_df
        .groupby(CLIENT_COL)["risk_contribution_points"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    rating_reasoning_df = rating_reasoning_df.sort_values([CLIENT_COL, "rating_reason_rank"])
    return cluster_rating_df, rating_base, weights_df, rating_reasoning_df


def run_dbscan_eps_search(X_scaled: np.ndarray, min_samples: int) -> pd.DataFrame:
    rows = []
    for eps in DBSCAN_EPS_SEARCH:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
        metrics = cluster_metrics(X_scaled, labels, f"DBSCAN_eps_{eps}")
        metrics["eps"] = eps
        metrics["min_samples"] = min_samples
        rows.append(metrics)
    return pd.DataFrame(rows)


print("=" * 65)
print("COMPONENTE 2 - CLUSTERING DE CLIENTES (COBRANZAS)")
print("=" * 65)

missing_inputs = [
    path for path in [CLIENT_FEATURES_CSV, CLIENT_FEATURE_LIST_CSV]
    if not path.exists()
]
if missing_inputs:
    missing_txt = "\n".join(f"  - {path}" for path in missing_inputs)
    sys.exit(
        "\n[ERROR FATAL] Faltan artefactos de preparacion para clustering:\n"
        f"{missing_txt}\n"
        "Ejecuta primero 03_preparacion/notebook_preparacion.ipynb.\n"
    )

client_features = pd.read_csv(CLIENT_FEATURES_CSV)
feature_cols = pd.read_csv(CLIENT_FEATURE_LIST_CSV)["feature"].dropna().astype(str).tolist()

missing_cols = [col for col in [CLIENT_COL, *feature_cols] if col not in client_features.columns]
if missing_cols:
    sys.exit(f"\n[ERROR FATAL] Columnas faltantes en client_features_clustering_base.csv: {missing_cols}\n")

client_ids = client_features[CLIENT_COL].astype(str).to_numpy()
X_raw, feature_names, log1p_cols = prepare_clustering_matrix(client_features, feature_cols)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw.values)

print(f"[OK] Dataset base de clientes cargado: {client_features.shape}")
print(f"[OK] Clientes analizados: {len(client_ids):,}")
print(f"[OK] Features para clustering: {len(feature_names)}")
print(f"[OK] log1p aplicado a {len(log1p_cols)} columnas agregadas.")

pd.DataFrame({"feature": feature_names}).to_csv(
    OUTPUT_DIR / "clustering_model_features.csv",
    index=False,
)

print("\n[STEP] Busqueda auxiliar de k optimo (silhouette) ...")
k_search_rows = []
for k in KMEANS_K_SEARCH:
    if k >= len(client_ids):
        break
    km_tmp = KMeans(n_clusters=k, random_state=KMEANS_SEED, n_init=10)
    labels_tmp = km_tmp.fit_predict(X_scaled)
    sil_tmp = silhouette_score(X_scaled, labels_tmp) if len(set(labels_tmp)) > 1 else -1
    metrics_tmp = cluster_metrics(X_scaled, labels_tmp, f"KMeans_k{k}_search")
    k_search_rows.append({
        "method": f"KMeans_k{k}_search",
        "k_evaluated": k,
        "inertia": km_tmp.inertia_,
        "silhouette": sil_tmp,
        "davies_bouldin": metrics_tmp["davies_bouldin"],
        "calinski_harabasz": metrics_tmp["calinski_harabasz"],
        "noise_ratio": metrics_tmp["noise_ratio"],
        "n_noise": metrics_tmp["n_noise"],
        "n_clusters": k,
    })
    print(f"  k={k:2d} | inertia={km_tmp.inertia_:.2f} | silhouette={sil_tmp:.4f}")

save_k_search_plot(k_search_rows)
best_k_sil = max(k_search_rows, key=lambda row: row["silhouette"])["k_evaluated"] if k_search_rows else KMEANS_K_FIXED
print(f"[INFO] k optimo por silhouette: {best_k_sil} | k principal del proyecto: {KMEANS_K_FIXED}")

print(f"\n[STEP] KMeans principal con k={KMEANS_K_FIXED} ...")
kmeans_main = KMeans(n_clusters=KMEANS_K_FIXED, random_state=KMEANS_SEED, n_init=20)
km_labels = kmeans_main.fit_predict(X_scaled)
km_metrics = cluster_metrics(X_scaled, km_labels, f"KMeans_k{KMEANS_K_FIXED}")

print(f"  Silhouette:        {km_metrics['silhouette']:.4f}")
print(f"  Davies-Bouldin:    {km_metrics['davies_bouldin']:.4f}")
print(f"  Calinski-Harabasz: {km_metrics['calinski_harabasz']:.4f}")

km_labels_opt = None
km_metrics_opt = {}
if best_k_sil != KMEANS_K_FIXED:
    print(f"\n[STEP] KMeans auxiliar con k={best_k_sil} (k optimo silhouette) ...")
    kmeans_opt = KMeans(n_clusters=best_k_sil, random_state=KMEANS_SEED, n_init=20)
    km_labels_opt = kmeans_opt.fit_predict(X_scaled)
    km_metrics_opt = cluster_metrics(X_scaled, km_labels_opt, f"KMeans_k{best_k_sil}_optimal")
    print(f"  Silhouette:        {km_metrics_opt['silhouette']:.4f}")

effective_min = min(DBSCAN_MIN_SAMPLES, max(2, len(client_ids) // 20))
print(f"\n[STEP] DBSCAN (eps={DBSCAN_EPS}, min_samples={effective_min}) ...")
dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=effective_min)
db_labels = dbscan.fit_predict(X_scaled)
db_metrics = cluster_metrics(X_scaled, db_labels, "DBSCAN")

print(f"  Clusters:          {db_metrics['n_clusters']}")
print(f"  Ruido:             {db_metrics['n_noise']} ({db_metrics['noise_ratio']:.2%})")
if not np.isnan(db_metrics["silhouette"]):
    print(f"  Silhouette:        {db_metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin:    {db_metrics['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz: {db_metrics['calinski_harabasz']:.4f}")

print("\n[STEP] Probando sensibilidad de DBSCAN a eps ...")
dbscan_eps_df = run_dbscan_eps_search(X_scaled, effective_min)
dbscan_eps_path = OUTPUT_DIR / "dbscan_eps_search.csv"
dbscan_eps_df.to_csv(dbscan_eps_path, index=False)
for _, row in dbscan_eps_df.iterrows():
    print(
        f"  eps={row['eps']:<4} | clusters={int(row['n_clusters']):>2} "
        f"| ruido={row['noise_ratio']:.1%}"
    )
print(f"[SAVED] {dbscan_eps_path}")

metrics_rows = [km_metrics, db_metrics]
if km_metrics_opt:
    metrics_rows.append(km_metrics_opt)
metrics_rows += k_search_rows

clustering_metrics_df = pd.DataFrame(metrics_rows)
if "k_evaluated" not in clustering_metrics_df.columns:
    clustering_metrics_df["k_evaluated"] = np.nan
else:
    clustering_metrics_df["k_evaluated"] = clustering_metrics_df["k_evaluated"].fillna(
        clustering_metrics_df["n_clusters"]
    )

metrics_path = OUTPUT_DIR / "clustering_metrics.csv"
clustering_metrics_df.to_csv(metrics_path, index=False)
print(f"\n[SAVED] {metrics_path}")

assignments_df = pd.DataFrame({
    CLIENT_COL: client_ids,
    f"cluster_kmeans_k{KMEANS_K_FIXED}": km_labels,
    "cluster_dbscan": db_labels,
})
if km_labels_opt is not None:
    assignments_df[f"cluster_kmeans_k{best_k_sil}_optimal"] = km_labels_opt

assignments_path = OUTPUT_DIR / "client_cluster_assignments.csv"
assignments_df.to_csv(assignments_path, index=False)
print(f"[SAVED] {assignments_path}")

print("\n[STEP] Calculando perfiles por cluster ...")
profile_base = client_features[[CLIENT_COL, *feature_names]].copy()
profile_base["cluster_kmeans"] = km_labels

profile_df = (
    profile_base
    .groupby("cluster_kmeans")[feature_names]
    .mean()
    .reset_index()
    .rename(columns={"cluster_kmeans": "cluster"})
)

sizes = pd.Series(km_labels).value_counts().sort_index().reset_index()
sizes.columns = ["cluster", "n_clientes"]
profile_df = profile_df.merge(sizes, on="cluster", how="left")

if SECTOR_PROFILE_COL in client_features.columns:
    sector_tmp = pd.DataFrame({
        CLIENT_COL: client_ids,
        "cluster_kmeans": km_labels,
        SECTOR_PROFILE_COL: client_features[SECTOR_PROFILE_COL].astype(str),
    })
    sector_by_cluster = (
        sector_tmp.groupby("cluster_kmeans")[SECTOR_PROFILE_COL]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A")
        .reset_index()
        .rename(columns={"cluster_kmeans": "cluster"})
    )
    profile_df = profile_df.merge(sector_by_cluster, on="cluster", how="left")
    print("[OK] Sector dominante modal agregado a perfiles.")

front_cols = ["cluster", "n_clientes"]
if SECTOR_PROFILE_COL in profile_df.columns:
    front_cols.append(SECTOR_PROFILE_COL)
other_cols = [col for col in profile_df.columns if col not in front_cols]
profile_df = profile_df[front_cols + other_cols]

profiles_path = OUTPUT_DIR / "cluster_profiles.csv"
profile_df.to_csv(profiles_path, index=False)
print(f"[SAVED] {profiles_path}")

print("\n[STEP] Generando explicabilidad visual y por cliente ...")
drivers_df = build_feature_drivers(profile_base, feature_names, kmeans_main.cluster_centers_)
drivers_path = OUTPUT_DIR / "cluster_feature_drivers.csv"
drivers_df.to_csv(drivers_path, index=False)

top_drivers_path = OUTPUT_DIR / "cluster_top_features.csv"
drivers_df[drivers_df["rank_in_cluster"] <= TOP_DRIVER_FEATURES].to_csv(
    top_drivers_path,
    index=False,
)

readable_summary_df = build_cluster_readable_summary(profile_df, drivers_df, len(client_ids))
readable_summary_path = OUTPUT_DIR / "cluster_readable_summary.csv"
readable_summary_df.to_csv(readable_summary_path, index=False)

assignments_detailed_df, examples_df = build_client_distance_tables(
    X_scaled,
    km_labels,
    client_ids,
    kmeans_main.cluster_centers_,
    client_features,
)
detailed_assignments_path = OUTPUT_DIR / "client_cluster_assignments_detailed.csv"
examples_path = OUTPUT_DIR / "client_cluster_examples.csv"
assignments_detailed_df.to_csv(detailed_assignments_path, index=False)
examples_df.to_csv(examples_path, index=False)

client_reasoning_df = build_client_reasoning(
    client_features[[CLIENT_COL, *feature_names]],
    assignments_detailed_df,
    drivers_df,
)
client_reasoning_path = OUTPUT_DIR / "client_cluster_reasoning.csv"
client_reasoning_df.to_csv(client_reasoning_path, index=False)

cluster_rating_df, client_rating_df, rating_weights_df, rating_reasoning_df = build_cluster_ratings(
    client_features[[CLIENT_COL, *feature_names, SECTOR_PROFILE_COL]]
    if SECTOR_PROFILE_COL in client_features.columns
    else client_features[[CLIENT_COL, *feature_names]],
    assignments_detailed_df,
)
cluster_rating_path = OUTPUT_DIR / "cluster_star_ratings.csv"
client_rating_path = OUTPUT_DIR / "client_star_ratings.csv"
rating_weights_path = OUTPUT_DIR / "star_rating_feature_weights.csv"
rating_reasoning_path = OUTPUT_DIR / "client_star_rating_reasoning.csv"
frontend_view_path = OUTPUT_DIR / "frontend_customer_segments.csv"

readable_summary_df = readable_summary_df.merge(
    cluster_rating_df[["cluster", "tipo_cliente", "cluster_risk_score_0_100", "rating_cluster_promedio"]],
    on="cluster",
    how="left",
)
readable_summary_df.to_csv(readable_summary_path, index=False)

frontend_view_df = build_frontend_customer_view(
    client_rating_df,
    cluster_rating_df,
    client_reasoning_df,
    rating_reasoning_df,
)

cluster_rating_df.to_csv(cluster_rating_path, index=False)
client_rating_df.to_csv(client_rating_path, index=False)
rating_weights_df.to_csv(rating_weights_path, index=False)
rating_reasoning_df.to_csv(rating_reasoning_path, index=False)
frontend_view_df.to_csv(frontend_view_path, index=False)

pca_df = save_pca_plot(X_scaled, km_labels, client_ids, client_features)
save_cluster_size_plot(profile_df, len(client_ids))
save_driver_heatmap(drivers_df)
save_key_profile_heatmap(profile_df, feature_names)

print(f"[SAVED] {drivers_path}")
print(f"[SAVED] {top_drivers_path}")
print(f"[SAVED] {readable_summary_path}")
print(f"[SAVED] {detailed_assignments_path}")
print(f"[SAVED] {examples_path}")
print(f"[SAVED] {client_reasoning_path}")
print(f"[SAVED] {cluster_rating_path}")
print(f"[SAVED] {client_rating_path}")
print(f"[SAVED] {rating_weights_path}")
print(f"[SAVED] {rating_reasoning_path}")
print(f"[SAVED] {frontend_view_path}")
print(f"[SAVED] {OUTPUT_DIR / 'cluster_pca_coordinates.csv'}")
print(f"[SAVED] {OUTPUT_DIR / 'clustering_k_search.png'}")
print(f"[SAVED] {OUTPUT_DIR / 'cluster_pca_scatter.png'}")
print(f"[SAVED] {OUTPUT_DIR / 'cluster_sizes.png'}")
print(f"[SAVED] {OUTPUT_DIR / 'cluster_feature_drivers_heatmap.png'}")
print(f"[SAVED] {OUTPUT_DIR / 'cluster_key_profile_heatmap.png'}")

print("\n" + "=" * 65)
print("RESUMEN - COMPONENTE 2 CLUSTERING DE CLIENTES")
print("=" * 65)
print(f"  Dataset fuente:            {CLIENT_FEATURES_CSV.relative_to(PROJECT_DIR)}")
print(f"  Clientes analizados:       {len(client_ids):,}")
print(f"  Features de comportamiento: {len(feature_names)}")
print(f"  Log1p aplicado a:          {len(log1p_cols)} columnas")
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
    print("    (coincide con k principal)")
print("")
print("  DBSCAN")
print(f"    Clusters:                {db_metrics['n_clusters']}")
print(f"    Noise ratio:             {db_metrics['noise_ratio']:.2%}")
sil_db = db_metrics["silhouette"]
print(f"    Silhouette (sin ruido):  {sil_db:.4f}" if not np.isnan(sil_db) else "    Silhouette:              N/A (clusters insuficientes)")

print("\n  Distribucion de clientes por cluster (KMeans principal):")
for _, row in profile_df[["cluster", "n_clientes"]].iterrows():
    pct = row["n_clientes"] / len(client_ids) * 100
    bar = "#" * int(pct / 2)
    print(f"    Cluster {int(row['cluster'])}: {int(row['n_clientes']):>5} clientes ({pct:5.1f}%) {bar}")

print("\n  Archivos exportados:")
for fname in [
    "clustering_metrics.csv",
    "client_cluster_assignments.csv",
    "cluster_profiles.csv",
    "clustering_model_features.csv",
    "cluster_feature_drivers.csv",
    "cluster_top_features.csv",
    "cluster_readable_summary.csv",
    "client_cluster_assignments_detailed.csv",
    "client_cluster_examples.csv",
    "client_cluster_reasoning.csv",
    "cluster_star_ratings.csv",
    "client_star_ratings.csv",
    "star_rating_feature_weights.csv",
    "client_star_rating_reasoning.csv",
    "frontend_customer_segments.csv",
    "cluster_pca_coordinates.csv",
    "dbscan_eps_search.csv",
    "clustering_k_search.png",
    "cluster_pca_scatter.png",
    "cluster_sizes.png",
    "cluster_feature_drivers_heatmap.png",
    "cluster_key_profile_heatmap.png",
]:
    path = OUTPUT_DIR / fname
    status = "OK" if path.exists() else "FALTANTE"
    print(f"    [{status}] {path}")

print("=" * 65)
print("[OK] Componente 2 completado exitosamente.")
