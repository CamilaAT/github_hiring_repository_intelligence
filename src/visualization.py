"""
Gráficos para el análisis exploratorio y la presentación de resultados.

Cada función devuelve una `matplotlib.figure.Figure` (para que la app de
Streamlit la renderice con `st.pyplot`) y, opcionalmente, la guarda en
`output/figures/`. Toda la lógica de cálculo vive en `evaluation.py`; aquí
solo se transforma a visualización.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation import (
    ALL_LABELS,
    NUMERIC_FEATURES,
    load_labeled,
    load_predictions,
    compute_confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "output" / "figures"

# Paleta consistente para las 6 categorías (de menor a mayor madurez + outliers)
LABEL_ORDER = ["intern", "junior", "senior", "lead", "template", "low_value"]
LABEL_COLORS = {
    "intern": "#9ecae1",
    "junior": "#4292c6",
    "senior": "#2171b5",
    "lead": "#08519c",
    "template": "#fdae6b",
    "low_value": "#bdbdbd",
}

sns.set_style("whitegrid")


def _save(fig: plt.Figure, filename: str | None) -> None:
    if filename is None:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")


# ──────────────────────────────────────────
# 1) Distribución de categorías
# ──────────────────────────────────────────

def plot_label_distribution(
    labeled: pd.DataFrame | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Conteo de repos por categoría — muestra el desbalance de clases."""
    labeled = load_labeled() if labeled is None else labeled
    counts = labeled["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[LABEL_COLORS[l] for l in counts.index],
        edgecolor="black",
        linewidth=0.6,
    )
    max_v = max(counts.values)
    for bar, v in zip(bars, counts.values):
        pct = v / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max_v * 0.025,
            f"{int(v)}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, linespacing=0.95,
        )

    ax.set_title(
        f"Distribución de categorías en el dataset (n={total})\n"
        "Tres clases dominan; tres son casi inexistentes",
        fontsize=12.5,
    )
    ax.set_ylabel("Cantidad de repositorios")
    ax.set_xlabel("Categoría asignada por DeepSeek (de menor a mayor madurez)")
    ax.set_ylim(0, max_v * 1.22)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ──────────────────────────────────────────
# 2) Señales promedio por categoría  (Q1: madurez)
# ──────────────────────────────────────────

def plot_signals_by_label(
    labeled: pd.DataFrame | None = None,
    features: list[str] | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Heatmap normalizado (z-score por columna) de la media de cada señal por clase."""
    labeled = load_labeled() if labeled is None else labeled
    if features is None:
        features = [
            "stars", "forks", "contributors", "merged_prs", "releases",
            "recent_commits", "commit_rate", "readme_length",
            "has_cicd", "has_license", "repo_age_days", "engagement_ratio",
        ]
    available = [c for c in features if c in labeled.columns]
    order = [c for c in LABEL_ORDER if c in labeled["label"].unique()]
    agg = labeled.groupby("label")[available].mean().loc[order]

    norm = (agg - agg.mean()) / agg.std(ddof=0).replace(0, 1)

    # Formato compacto para la anotación: 1 decimal para valores < 100, entero
    # con coma para mayores. Evita que cifras largas (p.ej. stars promedio)
    # invadan celdas vecinas y compriman las filas template / low_value.
    def _fmt(x: float) -> str:
        if pd.isna(x):
            return ""
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        if abs(x) >= 100:
            return f"{x:.0f}"
        if abs(x) >= 10:
            return f"{x:.1f}"
        return f"{x:.2f}"

    annot = agg.map(_fmt)

    # Figura más alta para que las 6 filas (incluidas template y low_value)
    # tengan espacio vertical suficiente y los ticks no se traslapen.
    fig, ax = plt.subplots(figsize=(13, 6.2))
    sns.heatmap(
        norm, annot=annot, fmt="", cmap="RdBu_r", center=0,
        cbar_kws={"label": "z-score por columna (rojo = sobre el promedio)"},
        linewidths=0.6, linecolor="white", ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title(
        "Q1 — Media de cada señal por categoría\n"
        "Color = z-score por columna (rojo = arriba del promedio del dataset). "
        "Número dentro de cada celda = valor real (no normalizado).",
        fontsize=12, pad=12,
    )
    ax.set_xlabel("Señal de GitHub", fontsize=11)
    ax.set_ylabel("Categoría", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
    # Margen extra abajo/izquierda para que ningún tick quede recortado
    fig.subplots_adjust(left=0.12, bottom=0.20, top=0.88, right=0.98)
    _save(fig, save_as)
    return fig


# ──────────────────────────────────────────
# 3) Bajo valor / template vs. real (Q2)
# ──────────────────────────────────────────

def plot_low_value_vs_real(
    labeled: pd.DataFrame | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """
    Q2 — ¿Cómo distinguir un repo template/low_value de uno con ingeniería real?

    Para responderlo, mostramos qué porcentaje de repos en cada cubeta cumple
    con cada práctica de ingeniería (binaria). La separación se ve "a ojo": las
    prácticas que tienen mucha mayor adopción en el bucket de ingeniería real
    son las que mejor discriminan.
    """
    labeled = load_labeled() if labeled is None else labeled
    df = labeled.copy()
    is_template = df["label"].isin(["template", "low_value"])
    n_tpl = int(is_template.sum())
    n_eng = int((~is_template).sum())

    label_eng = f"intern / junior / senior / lead  (n={n_eng})"
    label_tpl = f"template / low_value  (n={n_tpl})"
    df["bucket"] = np.where(is_template, label_tpl, label_eng)

    # Cada práctica es una condición binaria sobre las señales del repo
    practices = {
        "Multi-contribuidor (≥ 2)": df["contributors"] >= 2,
        "Tiene PRs merged (> 0)": df["merged_prs"] > 0,
        "CI/CD configurado": df["has_cicd"] == 1,
        "README detallado (> 500 chars)": df["readme_length"] > 500,
        "Licencia open source": df["has_license"] == 1,
        "Tiene releases (> 0)": df["releases"] > 0,
        "Topics etiquetados (> 0)": df["topics_count"] > 0,
        "Tiene descripción": df["has_description"] == 1,
    }

    rows = []
    for name, mask in practices.items():
        for bucket_name in [label_eng, label_tpl]:
            in_bucket = df["bucket"] == bucket_name
            pct = (mask & in_bucket).sum() / max(in_bucket.sum(), 1) * 100
            rows.append({"practice": name, "bucket": bucket_name, "pct": pct})
    out = pd.DataFrame(rows)

    # Ordenamos las prácticas por brecha (engineering − template) descendente:
    # arriba quedan las que más discriminan
    pivot = out.pivot(index="practice", columns="bucket", values="pct")
    pivot["gap"] = pivot[label_eng] - pivot[label_tpl]
    order = pivot.sort_values("gap", ascending=True).index.tolist()  # ascending → top mayor brecha

    eng_vals = pivot.loc[order, label_eng].values
    tpl_vals = pivot.loc[order, label_tpl].values

    y = np.arange(len(order))
    height = 0.38

    fig, ax = plt.subplots(figsize=(11.5, 6))
    b1 = ax.barh(y + height / 2, eng_vals, height,
                 color="#2171b5", edgecolor="black", linewidth=0.5,
                 label=label_eng)
    b2 = ax.barh(y - height / 2, tpl_vals, height,
                 color="#bdbdbd", edgecolor="black", linewidth=0.5,
                 label=label_tpl)
    for bar, v in zip(b1, eng_vals):
        ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}%", va="center", fontsize=9, color="#08306b")
    for bar, v in zip(b2, tpl_vals):
        ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}%", va="center", fontsize=9, color="#525252")

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_xlabel("% de repos en la cubeta que cumplen con la práctica", fontsize=11)
    ax.set_title(
        "Q2 — Prácticas de ingeniería: ¿qué tan presentes están en cada cubeta?\n"
        "Las prácticas están ordenadas de menor a mayor brecha (las de arriba son las que mejor discriminan).",
        fontsize=12, pad=10,
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ──────────────────────────────────────────
# 4) Matriz de confusión (heatmap)
# ──────────────────────────────────────────

def plot_confusion_matrix(
    predictions: pd.DataFrame | None = None,
    labels: list[str] = ALL_LABELS,
    save_as: str | None = None,
) -> plt.Figure:
    """Heatmap de la matriz de confusión del DistilBERT sobre el test set."""
    predictions = load_predictions() if predictions is None else predictions
    cm = compute_confusion_matrix(
        predictions["true_label"].tolist(),
        predictions["predicted_label"].tolist(),
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        cbar_kws={"label": "Conteo"}, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title("Matriz de confusión — DistilBERT en test set (n=72)")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ──────────────────────────────────────────
# 5) Baseline vs BERT por clase (Q4)
# ──────────────────────────────────────────

def plot_baseline_vs_bert(
    comparison: pd.DataFrame,
    save_as: str | None = None,
) -> plt.Figure:
    """Comparación por clase del F1 de DistilBERT vs Regresión Logística."""
    per_class = comparison[(comparison["metric"] == "f1") &
                           (comparison["scope"].isin(ALL_LABELS))].copy()
    per_class["scope"] = pd.Categorical(per_class["scope"], categories=LABEL_ORDER, ordered=True)
    per_class = per_class.sort_values("scope")

    x = np.arange(len(per_class))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, per_class["bert"], width,
           label="DistilBERT (texto)", color="#08519c", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, per_class["baseline_lr"], width,
           label="Regresión Logística (señales)", color="#fd8d3c", edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(per_class["scope"], rotation=15, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.set_title("Q4 — DistilBERT vs baseline numérico por categoría")
    ax.legend(loc="upper right")

    for i, (b, l) in enumerate(zip(per_class["bert"], per_class["baseline_lr"])):
        ax.text(i - width / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, l + 0.02, f"{l:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ──────────────────────────────────────────
# 6) Distribución de confianza de DeepSeek
# ──────────────────────────────────────────

def plot_confidence_distribution(
    labeled: pd.DataFrame | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Histograma de confianza autoreportada por DeepSeek — útil para Tab 1/2."""
    labeled = load_labeled() if labeled is None else labeled
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=labeled, x="confidence", hue="label",
        multiple="stack", bins=20, palette=LABEL_COLORS, ax=ax,
        hue_order=[l for l in LABEL_ORDER if l in labeled["label"].unique()],
    )
    ax.set_title("Confianza autoreportada por DeepSeek por categoría")
    ax.set_xlabel("Confianza (0–1)")
    ax.set_ylabel("Repositorios")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ──────────────────────────────────────────
# CLI: regenerar todas las figuras
# ──────────────────────────────────────────

def regenerate_all_figures() -> None:
    """Guarda todas las figuras a `output/figures/`."""
    from src.evaluation import train_baseline

    labeled = load_labeled()
    predictions = load_predictions()
    baseline = train_baseline(labeled=labeled, bert_predictions=predictions)

    plot_label_distribution(labeled, save_as="01_label_distribution.png")
    plot_signals_by_label(labeled, save_as="02_signals_by_label.png")
    plot_low_value_vs_real(labeled, save_as="03_low_value_vs_real.png")
    plot_confusion_matrix(predictions, save_as="04_confusion_matrix.png")
    plot_baseline_vs_bert(baseline["comparison_table"], save_as="05_baseline_vs_bert.png")
    plot_confidence_distribution(labeled, save_as="06_confidence_distribution.png")
    print(f"Figuras guardadas en {FIGURES_DIR}")


if __name__ == "__main__":
    regenerate_all_figures()
