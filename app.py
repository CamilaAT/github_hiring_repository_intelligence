"""
Dashboard de Streamlit para el proyecto
`github_hiring_repository_intelligence` (Track A).

Cuatro pestañas obligatorias en el orden que pide el enunciado:
  1. Problema & Metodología
  2. Análisis Exploratorio
  3. Resultados del Modelo
  4. Exploración Interactiva

Toda la lógica de cálculo vive en `src/evaluation.py` y los gráficos en
`src/visualization.py`. Esta app sólo orquesta y presenta.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation import (
    ALL_LABELS,
    NUMERIC_FEATURES,
    class_imbalance_summary,
    compute_confusion_matrix,
    load_labeled,
    load_predictions,
    load_split,
    most_confused_pairs,
    misclassified_examples,
    signals_by_label,
    train_baseline,
    TRAIN_PATH, VAL_PATH, TEST_PATH,
)
from src.visualization import (
    plot_baseline_vs_bert,
    plot_confidence_distribution,
    plot_confusion_matrix,
    plot_label_distribution,
    plot_low_value_vs_real,
    plot_signals_by_label,
)

ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="GitHub Hiring Repository Intelligence — Track A",
    page_icon="📊",
    layout="wide",
)


# ──────────────────────────────────────────
# CARGA DE DATOS (cacheada)
# ──────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_all():
    labeled = load_labeled()
    predictions = load_predictions()
    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    test_df = load_split(TEST_PATH)
    return labeled, predictions, train_df, val_df, test_df


@st.cache_data(show_spinner=False)
def _baseline_cached():
    labeled, predictions, train_df, val_df, test_df = _load_all()
    return train_baseline(
        labeled=labeled,
        train_df=train_df, val_df=val_df, test_df=test_df,
        bert_predictions=predictions,
    )


labeled, predictions, train_df, val_df, test_df = _load_all()
baseline = _baseline_cached()


# ──────────────────────────────────────────
# CABECERA
# ──────────────────────────────────────────

st.title("📊 GitHub Hiring Repository Intelligence")
st.caption(
    "Track A · Clasificación de repositorios de GitHub por madurez de ingeniería "
    "(intern · junior · senior · lead · template · low_value) usando weak supervision "
    "(DeepSeek) + fine-tuning de DistilBERT."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Problema & Metodología",
    "Análisis Exploratorio",
    "Resultados del Modelo",
    "Exploración Interactiva",
])


# ──────────────────────────────────────────
# TAB 1 — PROBLEMA & METODOLOGÍA
# ──────────────────────────────────────────

with tab1:
    st.header("1. Problema & Metodología")

    st.subheader("Objetivo")
    st.markdown(
        """
        Construir un sistema de **inteligencia de hiring sobre repositorios públicos
        de GitHub** que, a partir de las señales que se pueden observar desde la
        API pública, sugiera un nivel de **madurez de ingeniería** para cada repo.

        El sistema es un pipeline de **weak supervision**:
        1. Un LLM (DeepSeek) etiqueta los 480 repos a partir de un resumen textual.
        2. Con esas etiquetas se hace **fine-tuning de DistilBERT** sobre el mismo
           resumen textual.
        3. Se compara DistilBERT contra un **baseline de Regresión Logística** sobre
           las señales numéricas (Q4 del Track A).
        """
    )

    st.subheader("Selección y construcción del dataset (480 repos)")
    st.markdown(
        """
        - Camila (Persona 1) recolectó **480 repositorios** vía la GitHub REST API,
          extrajo señales (`stars`, `forks`, `contributors`, `releases`, `merged_prs`,
          `has_cicd`, `readme_length`, `commit_rate`, `engagement_ratio`, etc.) y
          generó un **resumen textual** (`text_summary`) por repo que sintetiza esas
          señales en lenguaje natural — esa es la entrada de los modelos.
        - Mayra (Persona 2) etiquetó los 480 resúmenes con **DeepSeek** y luego
          hizo el **fine-tuning de DistilBERT** sobre la columna `text_summary`.
        - Split estratificado **70/15/15** (train 336, val 72, test 72). Las clases
          con menos de 5 muestras se agruparon temporalmente para que `stratify` no
          fallara, pero los splits siguen siendo aleatorios estratificados.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Repositorios totales", len(labeled))
        st.metric("Idiomas únicos", int(labeled["language"].nunique()))
    with c2:
        st.metric("Train / Val / Test", f"{len(train_df)} / {len(val_df)} / {len(test_df)}")
        st.metric("Categorías", len(ALL_LABELS))

    st.subheader("Señales de GitHub usadas")
    st.markdown(
        "Las siguientes señales alimentan tanto el resumen textual como el baseline numérico:"
    )
    st.code(", ".join(NUMERIC_FEATURES), language=None)

    st.subheader("Estrategia de prompt de DeepSeek")
    st.markdown(
        """
        El prompt del sistema le pide a DeepSeek que actúe como **recruiter técnico**
        y devuelva exclusivamente un JSON `{label, confidence, reasoning}` para una
        de seis categorías:

        - **intern** — proyectos de aprendizaje, solo dev, sin CI/CD, muy pocos commits.
        - **junior** — funcional con algo de estructura; documentación básica, testing limitado.
        - **senior** — buenas prácticas, CI/CD presente, varios contribuidores, releases regulares.
        - **lead** — proyecto complejo y maduro, equipos grandes, muchos PRs, releases múltiples.
        - **template** — boilerplate o starter kit con poco trabajo original.
        - **low_value** — abandonado, vacío, spam, sin contenido relevante.

        El reasoning autoreportado por DeepSeek queda guardado en
        `data/labeled/repos_labeled.csv` y se puede consultar en la pestaña 4.
        """
    )

    st.subheader("Modelo de clasificación")
    st.markdown(
        """
        - **Modelo:** `distilbert-base-uncased`, head de clasificación con 6 clases.
        - **Hiperparámetros:** `max_length=256`, `batch_size=16`, `epochs=3`,
          `lr=2e-5`, AdamW, warmup lineal del 10 %.
        - **Selección de modelo:** se guarda el checkpoint con menor val loss y se
          evalúa sobre el test set.
        - **Insumo textual:** la columna `text_summary` (no se usan las señales
          numéricas directamente — esa es la línea base de comparación de Q4).
        """
    )

    st.subheader("Limitaciones que el dataset hereda")
    st.markdown(
        """
        - **Dataset chico (n=480)** para 6 clases, con desbalance fuerte.
        - **Weak supervision:** DeepSeek puede equivocarse y BERT aprende esos
          errores. El máximo posible de accuracy queda acotado por la calidad de
          DeepSeek (cuya confianza autoreportada se ve abajo).
        - **Solo señales públicas observables:** no hay calidad de código, ni
          revisiones, ni cobertura de tests, ni contexto de empresa.
        """
    )

    st.pyplot(plot_confidence_distribution(labeled), width="stretch")


# ──────────────────────────────────────────
# TAB 2 — ANÁLISIS EXPLORATORIO
# ──────────────────────────────────────────

with tab2:
    st.header("2. Análisis Exploratorio")

    st.subheader("Distribución de categorías")
    st.pyplot(plot_label_distribution(labeled), width="stretch")
    st.markdown(
        "**¿Qué muestra?** El dataset está fuertemente desbalanceado: `lead`, "
        "`junior` y `senior` concentran ~97 % de los repos, mientras que `intern`, "
        "`template` y `low_value` casi no aparecen. Este desbalance es el factor "
        "que más limita el desempeño del modelo (ver pestaña 3)."
    )

    cls = class_imbalance_summary(labeled)
    st.dataframe(cls, hide_index=True, width="stretch")

    st.subheader("Q1 — ¿Qué señales separan los niveles de madurez?")
    st.pyplot(plot_signals_by_label(labeled), width="stretch")
    st.markdown(
        """
        **Cómo leerlo.** Cada celda muestra la **media** de la señal en cada categoría;
        el color es el z-score por columna (rojo = arriba del promedio, azul = abajo).

        - `lead` domina en `stars`, `forks`, `contributors`, `releases`, `merged_prs`
          y `readme_length`: es la firma del proyecto maduro con comunidad.
        - `senior` queda en una zona intermedia, suele tener CI/CD y licencia pero
          con menos comunidad que `lead`.
        - `junior` se distingue por **mucho `recent_commits`/`commit_rate`** pero
          poco stars/forks y pocos PRs merged — actividad alta sin tracción.
        - `intern` (n=2) es ruidoso por su tamaño minúsculo.
        """
    )

    st.subheader("Q2 — Distinguir templates / repos de bajo valor")
    st.pyplot(plot_low_value_vs_real(labeled), width="stretch")
    st.markdown(
        """
        **Cómo leerlo.** Comparamos las señales que más se asocian a complejidad real
        entre los repos `template` / `low_value` y el resto. Los repos sin valor
        ingenieril aparecen con **contributors ≈ 1**, **0 PRs merged**, README muy
        corto o ausente, y **CI/CD = 0**. Ese es el filtro práctico para descartar
        boilerplate antes de evaluar madurez.
        """
    )

    st.subheader("Estadísticas resumen del dataset")
    summary_features = [
        "stars", "forks", "contributors", "merged_prs", "releases",
        "recent_commits", "readme_length", "repo_age_days", "has_cicd",
    ]
    st.dataframe(
        labeled[summary_features].describe().round(2).T,
        width="stretch",
    )


# ──────────────────────────────────────────
# TAB 3 — RESULTADOS DEL MODELO
# ──────────────────────────────────────────

with tab3:
    st.header("3. Resultados del Modelo")

    bert_metrics = baseline["bert_metrics"]
    bm = baseline["baseline_metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy (BERT)", f"{bert_metrics['accuracy']:.1%}")
    c2.metric("F1 macro (BERT)", f"{bert_metrics['macro avg']['f1-score']:.2f}")
    c3.metric("F1 weighted (BERT)", f"{bert_metrics['weighted avg']['f1-score']:.2f}")
    c4.metric("Tamaño test set", int(bert_metrics["weighted avg"]["support"]))

    st.subheader("Métricas por categoría — DistilBERT")
    rows = []
    for lab in ALL_LABELS:
        m = bert_metrics.get(lab, {})
        rows.append({
            "categoría": lab,
            "precision": round(m.get("precision", 0), 3),
            "recall": round(m.get("recall", 0), 3),
            "f1-score": round(m.get("f1-score", 0), 3),
            "support": int(m.get("support", 0)),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Matriz de confusión")
    st.pyplot(plot_confusion_matrix(predictions), width="stretch")
    st.markdown(
        """
        **Lectura honesta.** El modelo acierta `lead` y `junior` (F1 ≈ 0.78–0.81)
        pero **colapsa `senior` hacia `lead` y `junior`** (recall de senior ≈ 0.10).
        Las clases minoritarias (`intern`, `template`, `low_value`) tienen 0–1
        muestras en test y el modelo simplemente nunca las predice — es **un
        artefacto del desbalance**, no de la arquitectura.
        """
    )

    st.subheader("Pares más confundidos")
    cm = compute_confusion_matrix(
        predictions["true_label"].tolist(),
        predictions["predicted_label"].tolist(),
    )
    st.dataframe(most_confused_pairs(cm, top_n=8), hide_index=True, width="stretch")

    st.subheader("Q4 — Sensibilidad metodológica: BERT vs baseline numérico")
    st.pyplot(plot_baseline_vs_bert(baseline["comparison_table"]),
              width="stretch")

    st.dataframe(baseline["comparison_table"], hide_index=True, width="stretch")

    st.markdown(
        f"""
        **Hallazgo central de Q4.** Una **Regresión Logística sobre las señales
        numéricas** (las mismas que están dentro del resumen textual) ya logra
        accuracy **{bm['accuracy']:.1%}** en el test set — comparable o superior
        al **{bert_metrics['accuracy']:.1%}** de DistilBERT —, y **rescata
        `senior` con F1 = {bm.get('senior', {}).get('f1-score', 0):.2f}** frente
        al **{bert_metrics.get('senior', {}).get('f1-score', 0):.2f}** del modelo
        de texto.

        Esto sugiere que:

        1. Las señales numéricas codifican casi toda la información discriminativa
           que el texto repite.
        2. DistilBERT con 336 ejemplos y clases desbalanceadas **no aprende a
           separar `senior` de `lead`/`junior`** y prefiere colapsar a las clases
           dominantes.
        3. **El cuello de botella no es el modelo, es el etiquetado:** mientras
           DeepSeek genere ≤ 5 muestras de las clases raras, ningún clasificador
           supervisado las va a aprender.
        """
    )

    with st.expander("Features más informativas en el baseline"):
        st.dataframe(
            baseline["feature_importance"].head(15),
            hide_index=True, width="stretch",
        )


# ──────────────────────────────────────────
# TAB 4 — EXPLORACIÓN INTERACTIVA
# ──────────────────────────────────────────

with tab4:
    st.header("4. Exploración Interactiva")
    st.caption("Busca, filtra y compara la predicción de BERT con la etiqueta de DeepSeek.")

    # Cruce: predicciones del test set + metadata completa del repo
    merged = predictions.merge(
        labeled[["repo_name", "language", "stars", "forks", "contributors",
                 "merged_prs", "releases", "has_cicd", "readme_length",
                 "commit_rate", "engagement_ratio", "topics",
                 "confidence", "reasoning"]],
        on="repo_name", how="left",
    )
    merged["correcto"] = merged["true_label"] == merged["predicted_label"]

    col1, col2, col3 = st.columns(3)
    with col1:
        category_filter = st.multiselect(
            "Filtrar por etiqueta REAL (DeepSeek)",
            options=ALL_LABELS,
            default=[],
        )
    with col2:
        prediction_filter = st.multiselect(
            "Filtrar por etiqueta PREDICHA (BERT)",
            options=ALL_LABELS,
            default=[],
        )
    with col3:
        correctness = st.radio(
            "Aciertos / errores",
            ["Todos", "Solo aciertos", "Solo errores"],
            horizontal=True,
        )

    search = st.text_input("Buscar por nombre de repo (substring, case-insensitive)")

    view = merged.copy()
    if category_filter:
        view = view[view["true_label"].isin(category_filter)]
    if prediction_filter:
        view = view[view["predicted_label"].isin(prediction_filter)]
    if correctness == "Solo aciertos":
        view = view[view["correcto"]]
    elif correctness == "Solo errores":
        view = view[~view["correcto"]]
    if search:
        view = view[view["repo_name"].str.contains(search, case=False, na=False)]

    st.caption(f"Mostrando **{len(view)}** repos del test set (n={len(merged)}).")

    st.dataframe(
        view[[
            "repo_name", "true_label", "predicted_label", "correcto",
            "stars", "forks", "contributors", "merged_prs", "releases",
            "has_cicd", "readme_length", "language", "confidence",
        ]].sort_values("correcto"),
        hide_index=True,
        width="stretch",
        height=380,
    )

    st.subheader("Ver detalle de un repo")
    if len(view) > 0:
        repo_choice = st.selectbox(
            "Selecciona un repo para ver su resumen textual y el reasoning de DeepSeek",
            options=view["repo_name"].tolist(),
        )
        row = view[view["repo_name"] == repo_choice].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Etiqueta real (DeepSeek)", row["true_label"])
        c2.metric("Predicción BERT", row["predicted_label"])
        c3.metric("¿Acierto?", "Sí ✅" if row["correcto"] else "No ❌")

        st.markdown(f"**Repositorio:** `{row['repo_name']}`")
        st.markdown(
            f"- ⭐ stars: **{int(row['stars'])}**  ·  "
            f"🍴 forks: **{int(row['forks'])}**  ·  "
            f"👥 contributors: **{int(row['contributors'])}**  ·  "
            f"🔁 merged PRs: **{int(row['merged_prs'])}**  ·  "
            f"🏷️ releases: **{int(row['releases'])}**"
        )
        st.markdown(
            f"- CI/CD: **{'sí' if row['has_cicd'] else 'no'}**  ·  "
            f"README: **{int(row['readme_length'])} chars**  ·  "
            f"lenguaje: **{row['language'] if pd.notna(row['language']) else '—'}**  ·  "
            f"confianza DeepSeek: **{row['confidence']:.2f}**"
        )

        with st.expander("Resumen textual usado por el modelo (`text_summary`)"):
            st.write(row["text_summary"])

        with st.expander("Reasoning autoreportado por DeepSeek"):
            st.write(row["reasoning"])
    else:
        st.info("Ningún repo coincide con los filtros actuales.")

    st.subheader("Ejemplos de errores (mismos repos del análisis del Tab 3)")
    examples = misclassified_examples(predictions, n_per_pair=2)
    st.dataframe(
        examples[["repo_name", "true_label", "predicted_label", "text_summary"]],
        hide_index=True, width="stretch",
    )
