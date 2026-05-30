"""
Evaluación del clasificador de madurez de ingeniería.

Carga las predicciones de DistilBERT sobre el test split, recalcula métricas
(accuracy, precision, recall, F1 por clase y macro/weighted), construye la
matriz de confusión, identifica los pares de clases más confundidos y entrena
un baseline de Regresión Logística sobre las señales numéricas para responder
la pregunta de sensibilidad metodológica del Track A (Q4).

Funciones pensadas para ser importadas por la app de Streamlit; el bloque
`if __name__ == "__main__"` deja todos los artefactos en `output/metrics/` y
`output/tables/`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Rutas relativas a la raíz del repo
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
METRICS_DIR = OUTPUT / "metrics"
TABLES_DIR = OUTPUT / "tables"

PREDICTIONS_PATH = DATA / "splits" / "test_predictions.csv"
LABELED_PATH = DATA / "labeled" / "repos_labeled.csv"
TRAIN_PATH = DATA / "splits" / "train.csv"
VAL_PATH = DATA / "splits" / "val.csv"
TEST_PATH = DATA / "splits" / "test.csv"
WITH_SUMMARY_PATH = DATA / "processed" / "repos_with_summary.csv"

# Las 6 clases del problema (label_classes.json del entrenamiento)
ALL_LABELS = ["intern", "junior", "lead", "low_value", "senior", "template"]

# Señales numéricas/booleanas disponibles para el baseline
NUMERIC_FEATURES = [
    "stars", "forks", "watchers", "open_issues", "size_kb",
    "repo_age_days", "last_commit_days", "releases", "contributors",
    "recent_commits", "merged_prs", "readme_length",
    "commit_rate", "engagement_ratio",
    "description_length", "topics_count",
    "has_cicd", "is_fork", "is_template", "is_archived",
    "has_license", "has_wiki", "has_pages", "has_description",
    "is_mature", "is_active",
]


# ──────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────

def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Carga las predicciones de DistilBERT sobre el test set."""
    return pd.read_csv(path, encoding="utf-8-sig")


def load_labeled(path: Path = LABELED_PATH) -> pd.DataFrame:
    """Carga el dataset completo con etiquetas y razonamientos de DeepSeek."""
    return pd.read_csv(path, encoding="utf-8-sig")


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


# ──────────────────────────────────────────
# MÉTRICAS DEL MODELO BERT
# ──────────────────────────────────────────

def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = ALL_LABELS,
) -> dict:
    """Devuelve un dict con accuracy global y métricas por clase + agregadas."""
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    report["accuracy"] = float(accuracy_score(y_true, y_pred))
    return report


def compute_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = ALL_LABELS,
) -> pd.DataFrame:
    """Matriz de confusión como DataFrame con etiquetas en filas/columnas."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


# ──────────────────────────────────────────
# ANÁLISIS DE ERRORES
# ──────────────────────────────────────────

def most_confused_pairs(cm_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Devuelve los pares (real, predicho) con más confusión (excluye la diagonal)."""
    rows = []
    for true_label in cm_df.index:
        for pred_label in cm_df.columns:
            if true_label == pred_label:
                continue
            count = int(cm_df.loc[true_label, pred_label])
            if count > 0:
                rows.append({
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": count,
                })
    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def misclassified_examples(
    predictions: pd.DataFrame,
    n_per_pair: int = 2,
) -> pd.DataFrame:
    """Devuelve ejemplos concretos de repos mal clasificados (texto + etiquetas)."""
    errors = predictions[predictions["true_label"] != predictions["predicted_label"]].copy()
    if errors.empty:
        return errors
    return (
        errors.groupby(["true_label", "predicted_label"], group_keys=False)
              .head(n_per_pair)
              .reset_index(drop=True)
    )


def class_imbalance_summary(labeled: pd.DataFrame) -> pd.DataFrame:
    """Conteos y porcentajes por clase en el dataset completo (las 480 etiquetas)."""
    counts = labeled["label"].value_counts().reindex(ALL_LABELS, fill_value=0)
    df = pd.DataFrame({
        "label": counts.index,
        "count": counts.values,
        "share_pct": (counts.values / counts.values.sum() * 100).round(2),
    })
    return df


# ──────────────────────────────────────────
# BASELINE: REGRESIÓN LOGÍSTICA SOBRE SEÑALES NUMÉRICAS (Q4)
# ──────────────────────────────────────────

def _attach_features(
    split_df: pd.DataFrame,
    labeled: pd.DataFrame,
) -> pd.DataFrame:
    """Cruza un split (text_summary, label_id) con las señales numéricas del repo."""
    cols = ["text_summary"] + NUMERIC_FEATURES + ["label"]
    enriched = (
        split_df.merge(
            labeled[cols].drop_duplicates(subset="text_summary"),
            on="text_summary",
            how="left",
        )
    )
    return enriched


def train_baseline(
    labeled: pd.DataFrame | None = None,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    bert_predictions: pd.DataFrame | None = None,
) -> dict:
    """
    Entrena una Regresión Logística (con scaling y class_weight='balanced')
    sobre las señales numéricas usando los mismos splits del DistilBERT.
    Devuelve métricas y una tabla comparativa BERT vs baseline.
    """
    labeled = load_labeled() if labeled is None else labeled
    train_df = load_split(TRAIN_PATH) if train_df is None else train_df
    val_df = load_split(VAL_PATH) if val_df is None else val_df
    test_df = load_split(TEST_PATH) if test_df is None else test_df
    bert_predictions = load_predictions() if bert_predictions is None else bert_predictions

    train_full = _attach_features(train_df, labeled)
    val_full = _attach_features(val_df, labeled)
    test_full = _attach_features(test_df, labeled)

    # train + val para el ajuste (val no se usó para tuning del baseline, así que
    # entrenar con la unión es justo y maximiza el aprendizaje del modelo lineal)
    fit_df = pd.concat([train_full, val_full], ignore_index=True)

    X_fit = fit_df[NUMERIC_FEATURES].astype(float).values
    y_fit = fit_df["label"].values
    X_test = test_full[NUMERIC_FEATURES].astype(float).values
    y_test = test_full["label"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])
    pipe.fit(X_fit, y_fit)
    y_pred_baseline = pipe.predict(X_test)

    baseline_metrics = compute_classification_metrics(
        y_test.tolist(), y_pred_baseline.tolist()
    )

    bert_metrics = compute_classification_metrics(
        bert_predictions["true_label"].tolist(),
        bert_predictions["predicted_label"].tolist(),
    )

    # Tabla side-by-side
    rows = []
    rows.append({
        "metric": "accuracy",
        "scope": "global",
        "bert": round(bert_metrics["accuracy"], 4),
        "baseline_lr": round(baseline_metrics["accuracy"], 4),
    })
    rows.append({
        "metric": "f1",
        "scope": "macro avg",
        "bert": round(bert_metrics["macro avg"]["f1-score"], 4),
        "baseline_lr": round(baseline_metrics["macro avg"]["f1-score"], 4),
    })
    rows.append({
        "metric": "f1",
        "scope": "weighted avg",
        "bert": round(bert_metrics["weighted avg"]["f1-score"], 4),
        "baseline_lr": round(baseline_metrics["weighted avg"]["f1-score"], 4),
    })
    for lab in ALL_LABELS:
        rows.append({
            "metric": "f1",
            "scope": lab,
            "bert": round(bert_metrics.get(lab, {}).get("f1-score", 0.0), 4),
            "baseline_lr": round(baseline_metrics.get(lab, {}).get("f1-score", 0.0), 4),
        })
    comparison = pd.DataFrame(rows)

    # Predicciones del baseline alineadas con el test split
    baseline_preds = pd.DataFrame({
        "text_summary": test_full["text_summary"].values,
        "true_label": y_test,
        "baseline_predicted_label": y_pred_baseline,
    })

    return {
        "baseline_metrics": baseline_metrics,
        "bert_metrics": bert_metrics,
        "comparison_table": comparison,
        "baseline_predictions": baseline_preds,
        "feature_importance": _feature_importance(pipe, NUMERIC_FEATURES),
    }


def _feature_importance(pipe: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Magnitud media de los coeficientes por feature como proxy de importancia."""
    lr: LogisticRegression = pipe.named_steps["lr"]
    # Promediamos el valor absoluto de los coeficientes sobre todas las clases
    mean_abs = np.mean(np.abs(lr.coef_), axis=0)
    out = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_coef": mean_abs,
    }).sort_values("mean_abs_coef", ascending=False).reset_index(drop=True)
    return out


# ──────────────────────────────────────────
# AGREGADOS Y SALVADO DE ARTEFACTOS
# ──────────────────────────────────────────

def signals_by_label(
    labeled: pd.DataFrame | None = None,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Media de cada señal numérica por clase — apoya las respuestas de Q1 y Q2."""
    labeled = load_labeled() if labeled is None else labeled
    features = NUMERIC_FEATURES if features is None else features
    agg = labeled.groupby("label")[features].mean().round(3)
    # Reordena para que las clases salgan en orden de madurez
    order = [c for c in ["intern", "junior", "senior", "lead", "template", "low_value"]
             if c in agg.index]
    return agg.loc[order]


def run_full_evaluation(save: bool = True) -> dict:
    """Pipeline completo: métricas BERT, errores, baseline LR y persistencia."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    labeled = load_labeled()

    bert_metrics = compute_classification_metrics(
        predictions["true_label"].tolist(),
        predictions["predicted_label"].tolist(),
    )
    cm = compute_confusion_matrix(
        predictions["true_label"].tolist(),
        predictions["predicted_label"].tolist(),
    )
    confused = most_confused_pairs(cm, top_n=8)
    errors = misclassified_examples(predictions, n_per_pair=2)
    imbalance = class_imbalance_summary(labeled)
    by_label = signals_by_label(labeled)

    baseline = train_baseline(
        labeled=labeled,
        bert_predictions=predictions,
    )

    if save:
        with open(METRICS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(bert_metrics, f, indent=2, ensure_ascii=False)

        with open(METRICS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(baseline["baseline_metrics"], f, indent=2, ensure_ascii=False)

        cm.to_csv(TABLES_DIR / "confusion_matrix.csv", encoding="utf-8-sig")
        confused.to_csv(TABLES_DIR / "confused_pairs.csv", index=False, encoding="utf-8-sig")
        errors.to_csv(TABLES_DIR / "error_examples.csv", index=False, encoding="utf-8-sig")
        imbalance.to_csv(TABLES_DIR / "class_imbalance.csv", index=False, encoding="utf-8-sig")
        by_label.to_csv(TABLES_DIR / "signals_by_label.csv", encoding="utf-8-sig")
        baseline["comparison_table"].to_csv(
            TABLES_DIR / "baseline_vs_bert.csv", index=False, encoding="utf-8-sig"
        )
        baseline["baseline_predictions"].to_csv(
            TABLES_DIR / "baseline_predictions.csv", index=False, encoding="utf-8-sig"
        )
        baseline["feature_importance"].to_csv(
            TABLES_DIR / "baseline_feature_importance.csv", index=False, encoding="utf-8-sig"
        )

    return {
        "bert_metrics": bert_metrics,
        "confusion_matrix": cm,
        "confused_pairs": confused,
        "error_examples": errors,
        "class_imbalance": imbalance,
        "signals_by_label": by_label,
        **baseline,
    }


if __name__ == "__main__":
    print("Ejecutando evaluación completa…")
    out = run_full_evaluation(save=True)

    print("\n=== Métricas DistilBERT (recalculadas desde test_predictions.csv) ===")
    print(f"Accuracy: {out['bert_metrics']['accuracy']:.4f}")
    for lab in ALL_LABELS:
        m = out["bert_metrics"].get(lab, {})
        print(f"  {lab:<10} F1={m.get('f1-score', 0):.3f}  "
              f"support={int(m.get('support', 0))}")

    print("\n=== Pares más confundidos ===")
    print(out["confused_pairs"].to_string(index=False))

    print("\n=== Desbalance de clases (dataset completo) ===")
    print(out["class_imbalance"].to_string(index=False))

    print("\n=== Comparación BERT vs baseline (Regresión Logística) ===")
    print(out["comparison_table"].to_string(index=False))

    print("\n=== Top features del baseline (|coef| medio) ===")
    print(out["feature_importance"].head(10).to_string(index=False))

    print("\nArtefactos guardados en output/metrics/ y output/tables/")
