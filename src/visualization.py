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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[LABEL_COLORS[l] for l in counts.index],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3, str(int(v)),
                ha="center", va="bottom", fontsize=10)

    ax.set_title("Distribución de categorías en el dataset (n=480)", fontsize=13)
    ax.set_ylabel("Cantidad de repositorios")
    ax.set_xlabel("Categoría asignada por DeepSeek")
    ax.set_ylim(0, max(counts.values) * 1.15)
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

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        norm, annot=agg.round(1), fmt="", cmap="RdBu_r", center=0,
        cbar_kws={"label": "z-score por columna"},
        linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_title(
        "Señales promedio por categoría (anotación = valor real; color = z-score)",
        fontsize=12,
    )
    ax.set_xlabel("Señal de GitHub")
    ax.set_ylabel("Categoría")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
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
    Compara las señales de complejidad real (contributors, merged_prs, readme,
    CI/CD) entre los repos catalogados como `template`/`low_value` y el resto.
    """
    labeled = load_labeled() if labeled is None else labeled
    df = labeled.copy()
    df["bucket"] = np.where(
        df["label"].isin(["template", "low_value"]),
        "template / low_value",
        "intern / junior / senior / lead",
    )
    features = [
        ("contributors", "Contribuidores"),
        ("merged_prs", "PRs merged"),
        ("readme_length", "README (chars)"),
        ("has_cicd", "Tiene CI/CD (0/1)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    palette = {"template / low_value": "#bdbdbd",
               "intern / junior / senior / lead": "#2171b5"}
    for ax, (col, title) in zip(axes, features):
        sns.boxplot(data=df, x="bucket", y=col, hue="bucket",
                    palette=palette, legend=False, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if col in {"merged_prs", "readme_length"}:
            ax.set_yscale("symlog")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)

    fig.suptitle(
        "¿Qué distingue a un repo template / low_value de uno con ingeniería real?",
        fontsize=13, y=1.02,
    )
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
