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
        El dataset se armó siguiendo estos pasos:

        1. **Recolección vía GitHub REST API.** Se recopilaron **480 repositorios
           públicos** buscando diversidad en lenguaje principal, rango de stars,
           edad del repo y nivel de actividad, para evitar sesgar el dataset hacia
           proyectos de un solo perfil.
        2. **Extracción de señales.** Por cada repo se calcularon **26 señales
           cuantitativas y booleanas**: `stars`, `forks`, `contributors`,
           `releases`, `merged_prs`, `has_cicd`, `readme_length`, `commit_rate`,
           `engagement_ratio`, `repo_age_days`, `is_mature`, `is_fork`,
           `is_template`, `has_license`, `has_wiki`, `has_pages`, etc.
        3. **Resumen textual (`text_summary`).** Las señales se convirtieron en un
           párrafo en lenguaje natural por repo (por ejemplo: *"medium-sized
           codebase. 89 stars (moderate visibility). 5 contributors (medium team).
           CI/CD configured…"*). Ese párrafo es la **única entrada del modelo**.
        4. **Weak labeling con DeepSeek.** Los 480 resúmenes se enviaron a DeepSeek
           con un prompt que le pide actuar como recruiter técnico y devolver una
           etiqueta entre seis categorías más un `confidence` y un `reasoning`.
        5. **Split estratificado 70/15/15** (train **336**, val **72**, test **72**,
           semilla 42). Las tres clases con menos de 5 muestras (`intern`,
           `template`, `low_value`) se agruparon temporalmente para que `stratify`
           no fallara; los splits siguen siendo aleatorios estratificados.
        6. **Fine-tuning de DistilBERT** sobre `text_summary` con las etiquetas
           débiles como ground truth y selección del checkpoint con menor val loss.
        7. **Evaluación.** Predicciones sobre el test set y comparación con un
           baseline de Regresión Logística entrenado sobre las 26 señales numéricas
           originales — la línea base de la pregunta de sensibilidad metodológica (Q4).
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

    st.markdown("#### Distribución de confianza autoreportada por DeepSeek")
    st.pyplot(plot_confidence_distribution(labeled), width="stretch")
    st.markdown(
        """
        **Tipo de gráfico:** histograma apilado.

        **Cómo leerlo:**
        - **Eje X** — confianza que DeepSeek se asignó a sí mismo al etiquetar
          cada repo (escala 0 a 1, donde 1 = totalmente seguro).
        - **Eje Y** — cantidad de repos en cada bin de confianza.
        - **Color** — categoría asignada (intern, junior, senior, lead, template,
          low_value).

        **Qué dice:** la mayoría de las etiquetas se concentran entre **0.80 y
        0.95**, con una media de **0.87**. DeepSeek casi nunca se reporta inseguro,
        lo cual es típico de los LLMs: tienden a no calibrar bien la incertidumbre.
        Esto es una **advertencia metodológica** — la confianza autoreportada no es
        un proxy confiable de la calidad real de la etiqueta y no debe usarse para
        pesar muestras durante el entrenamiento.
        """
    )


# ──────────────────────────────────────────
# TAB 2 — ANÁLISIS EXPLORATORIO
# ──────────────────────────────────────────

with tab2:
    st.header("2. Análisis Exploratorio")

    st.markdown(
        """
        Esta pestaña responde dos preguntas con un gráfico cada una:

        - **Distribución general** — ¿cómo se reparten los 480 repos entre las
          seis categorías?
        - **Q1** — ¿qué señales de GitHub separan los niveles de madurez
          (intern → junior → senior → lead)?
        - **Q2** — ¿cómo distinguir un repo `template`/`low_value` de uno con
          ingeniería real?

        Cada gráfico viene con su lectura paso a paso debajo.
        """
    )

    # ── Distribución de categorías
    st.subheader("Distribución de categorías en el dataset")
    st.pyplot(plot_label_distribution(labeled), width="stretch")
    st.markdown(
        """
        **Tipo de gráfico:** barras verticales.

        **Cómo leerlo:**
        - **Eje X** — las seis categorías ordenadas de menor a mayor madurez
          (intern → junior → senior → lead) más las dos categorías "fuera de
          escala" (template, low_value).
        - **Eje Y** — cantidad absoluta de repos en cada categoría.
        - **Anotación arriba de cada barra** — conteo absoluto y porcentaje del
          total (n=480).

        **Qué dice:** el dataset está **fuertemente desbalanceado**:
        `lead` (36.7 %), `junior` (31.5 %) y `senior` (29.4 %) concentran
        **~97.6 %** de los repos. Las tres clases minoritarias suman 12 repos
        en total — `intern` (n=2), `template` (n=2) y `low_value` (n=8) —, lo
        cual significa que **el modelo prácticamente no las verá durante el
        entrenamiento**. Este desbalance es la causa principal del bajo F1 en
        esas clases (ver pestaña 3) y motiva la discusión metodológica de Q4.
        """
    )

    cls = class_imbalance_summary(labeled)
    st.markdown("**Tabla detallada de la distribución:**")
    st.dataframe(cls, hide_index=True, width="stretch")

    # ── Q1 — heatmap
    st.subheader("Q1 — ¿Qué señales separan los niveles de madurez?")
    st.pyplot(plot_signals_by_label(labeled), width="stretch")
    st.markdown(
        """
        **Tipo de gráfico:** heatmap (mapa de calor) con anotación numérica.

        **Cómo leerlo:**
        - **Filas (eje Y)** — las seis categorías de madurez ordenadas de menor
          a mayor (intern arriba; lead, template y low_value abajo).
        - **Columnas (eje X)** — doce señales de ingeniería de GitHub.
        - **Color de la celda** — *z-score normalizado por columna*: rojo
          significa que la media de esa señal en esa categoría está **por encima
          del promedio del dataset**; azul, por debajo. Centrado en 0 (blanco).
        - **Número dentro de la celda** — el **valor real** (media de la señal
          en esa categoría), sin normalizar — para que el lector pueda
          interpretar magnitudes (p.ej. 589 stars promedio en `lead`).

        **Qué dice (respuesta a Q1):** se ve una **gradiente clara de madurez**
        de arriba hacia abajo en las primeras cuatro filas:

        - **`intern`** — todo en azul intenso: 0 stars, 0 forks, 0 PRs merged,
          0 releases. Es solo-dev sin tracción ni proceso.
        - **`junior`** — sigue azul en popularidad y proceso (≈70 stars, ≈2 PRs)
          pero con **`commit_rate` alto** (431 commits/mes en promedio): mucha
          actividad sin colaboración ni revisión.
        - **`senior`** — la fila se "tibia": 213 stars, 7.7 contribuidores,
          71 PRs merged, 22 releases, **CI/CD en 79 %** de los repos.
        - **`lead`** — toda la fila en rojo: 589 stars, 68 contribuidores,
          204 PRs merged, 125 releases, CI/CD en 90 % y licencia en 94 %.

        **`template` y `low_value`** rompen el ordenamiento de madurez: tienen
        valores intermedios en señales de actividad pero **cero o casi cero en
        PRs merged y releases** — son la firma de "actividad superficial sin
        ingeniería". El gráfico Q2 lo aterriza con prácticas binarias.

        Las señales que más discriminan madurez son entonces: **PRs merged,
        releases, contribuidores, CI/CD, licencia y readme_length**.
        """
    )

    # ── Q2 — engineering practices bar chart
    st.subheader("Q2 — ¿Cómo distinguir templates / repos de bajo valor de los reales?")
    st.pyplot(plot_low_value_vs_real(labeled), width="stretch")
    st.markdown(
        """
        **Tipo de gráfico:** barras horizontales agrupadas.

        **Cómo leerlo:**
        - **Filas (eje Y)** — ocho **prácticas de ingeniería binarias** (cada
          una se cumple o no se cumple). Están ordenadas de **mayor a menor
          brecha** entre las dos cubetas: las prácticas de arriba son las que
          mejor discriminan.
        - **Eje X** — porcentaje de repos en cada cubeta que cumplen con la
          práctica (0 a 100 %).
        - **Color** — *azul* = repos de ingeniería real
          (intern + junior + senior + lead, n=470). *Gris* = repos
          template/low_value (n=10).
        - **Cada práctica tiene una definición explícita** (≥ 2 contribuidores,
          > 0 PRs merged, etc.) para que la comparación sea inequívoca.

        **Qué dice (respuesta a Q2):** la frontera entre un repo "real" y uno
        boilerplate **no es la popularidad sino la presencia de proceso**.
        Las brechas más grandes están en:

        - **`Tiene PRs merged`** — 67 % en repos reales vs **10 % en
          template/low_value**: el flujo de revisión es la señal más fuerte de
          trabajo colaborativo.
        - **`Tiene releases`** — 50 % vs 10 %: formalizar versiones implica
          un mínimo de disciplina.
        - **`Licencia open source`** — 70 % vs 30 %: definir licencia es un
          marcador básico de intención de publicación.
        - **`CI/CD configurado`** — 63 % vs 30 %: automatización mínima.

        **Notar también** la práctica de abajo (`Tiene descripción`): aparece en
        100 % de template/low_value vs 82 % de los reales — los repos vacíos a
        menudo **sobrecompensan con descripciones**, pero eso por sí solo no
        es señal de ingeniería.

        **Regla operativa derivada** (también en el README):
        > Si un repo tiene `contributors ≤ 1` **y** `merged_prs == 0` **y**
        > `has_cicd == 0` **y** `readme_length < 500`, marcarlo como candidato
        > a template/low_value antes de evaluar madurez.
        """
    )

    # ── Estadísticas resumen
    st.subheader("Estadísticas resumen del dataset")
    st.markdown(
        "Tabla con la distribución estadística (mean, std, min, percentiles, max) "
        "de las principales señales numéricas a través de los 480 repos. Es útil "
        "para entender la **escala** y la **dispersión** de cada señal antes de "
        "interpretarla en los gráficos de arriba."
    )
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
        **Tipo de gráfico:** heatmap de matriz de confusión.

        **Cómo leerlo:**
        - **Filas (eje Y)** — etiqueta **real** del repo (asignada por DeepSeek).
        - **Columnas (eje X)** — etiqueta que **predijo DistilBERT**.
        - **Cada celda** muestra cuántos repos del test set (n=72) cayeron en esa
          combinación. La **diagonal principal** son los aciertos; todo lo
          que está fuera de la diagonal es un error.
        - **Color** — más oscuro indica más casos.

        **Qué dice:**
        - El modelo **acierta sólido `lead` (27/28) y `junior` (20/21)** — esas
          son las clases que dominan el train, por lo que aprende su patrón.
        - **Colapsa `senior` hacia `lead` y `junior`**: de 21 repos `senior`
          reales, predijo `lead` para 11 y `junior` para 8. Solo acertó 2.
          Eso explica el F1 de 0.17 en `senior`.
        - Para las clases minoritarias (`intern`, `template`, `low_value`) con
          0–1 muestras en test, **el modelo nunca las predice** — sus columnas
          están en blanco. Esto es un artefacto del **desbalance del train**,
          no de la arquitectura del modelo.
        """
    )

    st.subheader("Pares más confundidos")
    st.markdown(
        "Tabla que extrae los pares (etiqueta real → etiqueta predicha) con más "
        "errores en el test set. Ordenada por frecuencia descendente. Útil para "
        "ver el patrón de confusión sin tener que leer la matriz completa."
    )
    cm = compute_confusion_matrix(
        predictions["true_label"].tolist(),
        predictions["predicted_label"].tolist(),
    )
    st.dataframe(most_confused_pairs(cm, top_n=8), hide_index=True, width="stretch")

    st.subheader("Q4 — Sensibilidad metodológica: BERT vs baseline numérico")
    st.pyplot(plot_baseline_vs_bert(baseline["comparison_table"]),
              width="stretch")
    st.markdown(
        f"""
        **Tipo de gráfico:** barras agrupadas por categoría.

        **Cómo leerlo:**
        - **Eje X** — las seis categorías ordenadas de menor a mayor madurez.
        - **Eje Y** — F1-score por categoría (0 a 1, donde 1 es perfecto).
        - **Color azul** — DistilBERT fine-tuneado sobre el resumen textual.
        - **Color naranja** — Regresión Logística entrenada sobre las 26
          señales numéricas originales, con `class_weight='balanced'` y
          `StandardScaler`, evaluada en exactamente el mismo test set.

        **Qué dice (respuesta a Q4):** la barra naranja **iguala o supera** a la
        azul en casi todas las clases. La diferencia más dramática es en
        **`senior`**, donde el baseline pasa de 0.17 (BERT) a
        **{bm.get('senior', {}).get('f1-score', 0):.2f}**. Globalmente:
        accuracy {bm['accuracy']:.1%} vs {bert_metrics['accuracy']:.1%}; F1
        weighted **{bm['weighted avg']['f1-score']:.2f}** vs
        {bert_metrics['weighted avg']['f1-score']:.2f}.

        **Interpretación:**

        1. El `text_summary` es una **verbalización 1-a-1 de los números** —
           no aporta información nueva sobre las features tabulares.
        2. DistilBERT con 336 ejemplos y clases desbalanceadas **se "rinde"
           en `senior`** y predice las clases mayoritarias por defecto.
        3. La Regresión Logística, gracias a `class_weight='balanced'`, pesa
           las clases minoritarias y recupera `senior`.
        4. **El cuello de botella metodológico no es la arquitectura sino el
           etiquetado**: mientras DeepSeek genere ≤ 5 muestras de las clases
           raras, ningún clasificador supervisado las va a aprender bien.
        5. **Recomendación práctica:** en problemas chicos donde el texto es
           derivado de features tabulares, **probar baselines numéricos antes
           que LLMs grandes**.
        """
    )

    st.markdown("**Tabla detallada de la comparación:**")
    st.dataframe(baseline["comparison_table"], hide_index=True, width="stretch")

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
