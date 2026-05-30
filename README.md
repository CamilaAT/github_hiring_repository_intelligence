# GitHub Hiring Repository Intelligence

Proyecto de NLP/ML que clasifica repositorios públicos de GitHub en seis niveles de
**madurez de ingeniería** (`intern` · `junior` · `senior` · `lead` · `template` · `low_value`)
combinando señales de la API de GitHub, **weak labeling** con un LLM (DeepSeek) y
**fine-tuning de DistilBERT**.

- **Track elegido:** **Track A — Engineering Maturity Classifier.**
- **Tamaño del dataset:** 480 repositorios.
- **Repositorio:** <https://github.com/CamilaAT/github_hiring_repository_intelligence>
- **Enunciado:** <https://github.com/d2cml-ai/Data-Science-Python/issues/176>

---

## 1. Qué hace el proyecto

Dado un repositorio público, el sistema lo clasifica en una de seis categorías que
buscan resumir el **nivel de práctica de ingeniería** que el repo evidencia. La
hipótesis de negocio detrás del Track A es que un reclutador o un equipo técnico
puede usar este tipo de clasificación como **señal complementaria** durante el
screening — no como decisión final.

El pipeline tiene tres etapas:

1. **Recolección y resumen** (Persona 1, Camila) — descarga señales de la API de
   GitHub y construye un **resumen textual** (`text_summary`) por repo.
2. **Weak labeling + fine-tuning** (Persona 2, Mayra) — DeepSeek etiqueta los 480
   resúmenes y DistilBERT se fine-tunea sobre esas etiquetas débiles.
3. **Evaluación, baseline y dashboard** (Persona 3, Manuel) — métricas, matriz de
   confusión, baseline numérico para Q4 y app de Streamlit con cuatro pestañas.

---

## 2. Datos

### Cómo se eligieron los repos

Se recolectaron **480 repositorios** vía la GitHub REST API combinando búsquedas
por temas/lenguajes (`github_collector.py`) para asegurar diversidad: distintos
lenguajes principales, distintos rangos de stars, distintas edades.

### Señales de GitHub usadas

Por cada repo se extrajeron, entre otras:

`stars · forks · watchers · open_issues · size_kb · repo_age_days ·
last_commit_days · releases · contributors · recent_commits · merged_prs ·
has_cicd · readme_length · is_fork · is_template · is_archived · has_license ·
has_wiki · has_pages · has_description · commit_rate · engagement_ratio ·
is_mature · is_active · description_length · topics_count`

### Resúmenes textuales (`text_summary`)

`src/summarization.py` convierte las señales numéricas/booleanas en un párrafo
en lenguaje natural (p.ej. *"medium-sized codebase. 89 stars (moderate visibility).
10 forks (modest community interest). 5 contributors (medium team). 42 commits in
last 90 days (~14/month, moderate activity)…"*). Ese párrafo es la **única
entrada de DistilBERT** y también lo que ve DeepSeek al etiquetar.

### Prompt de DeepSeek

El system prompt (`src/llm_labeling.py`) le pide a DeepSeek que actúe como
recruiter técnico y devuelva exclusivamente un JSON
`{"label", "confidence", "reasoning"}` con una de las seis categorías.
La descripción de cada categoría es explícita (por ejemplo *intern* = solo dev,
sin CI/CD, muy pocos commits; *lead* = proyecto complejo y maduro con equipo
grande). Cada llamada se hace con `temperature=0.1` para que la salida sea
estable, y el reasoning se persiste para auditoría posterior (`reasoning` se
puede ver en la pestaña 4 de la app).

### Split

Estratificado **70 / 15 / 15** sobre las etiquetas de DeepSeek
(train **336**, val **72**, test **72**, semilla 42). Las clases con menos de
5 muestras (`intern`, `template`, `low_value`) se agruparon temporalmente para
que `stratify` no rompiera; el train sigue siendo aleatorio estratificado.

### Distribución de etiquetas (las 480)

| categoría   | n   | %     |
|-------------|----:|------:|
| lead        | 176 | 36.7% |
| junior      | 151 | 31.5% |
| senior      | 141 | 29.4% |
| low_value   | 8   |  1.7% |
| intern      | 2   |  0.4% |
| template    | 2   |  0.4% |

Este **desbalance severo** es el factor que más limita el resultado del modelo
(ver §4).

---

## 3. Modelo

- **Arquitectura:** `distilbert-base-uncased` con un head lineal de 6 clases.
- **Hiperparámetros:** `max_length=256`, `batch_size=16`, `epochs=3`,
  `lr=2e-5`, `AdamW`, warmup lineal del 10 %, gradient clipping a 1.0,
  semilla 42.
- **Selección de modelo:** se guarda el checkpoint con menor val loss y se
  evalúa sobre el test set.

---

## 4. Métricas finales

Recalculadas desde `data/splits/test_predictions.csv` (y coinciden con
`output/metrics/metrics.json`):

| categoría   | precision | recall | F1    | support |
|-------------|----------:|-------:|------:|--------:|
| intern      | 0.00      | 0.00   | 0.00  | 0       |
| junior      | 0.67      | 0.95   | **0.78** | 21    |
| lead        | 0.69      | 0.96   | **0.81** | 28    |
| low_value   | 0.00      | 0.00   | 0.00  | 1       |
| senior      | 0.67      | 0.10   | **0.17** | 21    |
| template    | 0.00      | 0.00   | 0.00  | 1       |
| **accuracy**       |       |        | **0.68** | 72  |
| macro avg   | 0.34      | 0.34   | 0.29  | 72      |
| weighted avg| 0.66      | 0.68   | 0.59  | 72      |

**Lectura honesta del resultado.** El modelo aprende bien `junior` y `lead`,
**colapsa `senior` hacia `lead`/`junior`** (11 + 8 confusiones de un total de 21)
y **nunca predice `intern`, `template`, ni `low_value`** porque en train hay
1–5 muestras de cada una. Es el patrón clásico del weak supervision con clases
desbalanceadas: el clasificador supervisado no puede aprender lo que el LLM
prácticamente no etiquetó.

### Baseline numérico (Q4)

Una **Regresión Logística** con `class_weight='balanced'` sobre las 26 señales
numéricas (entrenada con train + val, evaluada sobre el mismo test) llega a:

| escope          | DistilBERT | Logistic Regression |
|-----------------|-----------:|--------------------:|
| accuracy global | 0.68       | **0.69**            |
| F1 weighted     | 0.59       | **0.72**            |
| F1 macro        | 0.29       | **0.36**            |
| F1 `junior`     | **0.78**   | 0.69                |
| F1 `lead`       | 0.81       | **0.88**            |
| F1 `senior`     | 0.17       | **0.62**            |

El detalle vive en `output/tables/baseline_vs_bert.csv` y se ve graficado en
`output/figures/05_baseline_vs_bert.png` y en el tab 3 de la app.

---

## 5. Cómo correr el proyecto

### Requisitos

```bash
git clone https://github.com/CamilaAT/github_hiring_repository_intelligence
cd github_hiring_repository_intelligence
pip install -r requirements.txt
```

Para reproducir la recolección y el etiquetado se necesitan dos secretos en `.env`:

```
GITHUB_TOKEN=ghp_xxx
DEEPSEEK_API_KEY=sk-xxx
```

Si solo se quiere usar el dashboard, los CSV con datos ya etiquetados están en `data/`.

### Pipeline completo (ya está corrido — datos en `data/`)

```bash
# Persona 1 — recolección y resúmenes
python src/github_collector.py
python src/preprocessing.py
python src/summarization.py

# Persona 2 — weak labeling y fine-tuning
python src/llm_labeling.py
python src/train.py

# Persona 3 — evaluación + baseline + figuras
python src/evaluation.py
python -m src.visualization
```

### App de Streamlit

```bash
streamlit run app.py
```

La app levanta cuatro pestañas en el orden exacto que pide el enunciado:

1. **Problema & Metodología**
2. **Análisis Exploratorio**
3. **Resultados del Modelo**
4. **Exploración Interactiva**

---

## 6. Limitaciones principales

1. **Etiquetas débiles.** DeepSeek puede equivocarse y BERT hereda esos errores.
   La accuracy techo está acotada por la calidad del LLM.
2. **Dataset chico y desbalanceado.** 480 repos para 6 clases; tres clases
   tienen ≤ 8 muestras totales — no es realista esperar que BERT las aprenda.
3. **Solo señales públicas observables.** Nada de calidad de código, revisiones,
   coverage de tests, ni contexto privado o de empresa.
4. **El `text_summary` es una traducción 1-a-1 de los números.** Por eso un
   modelo lineal sobre las features originales empata con o supera a BERT —
   no hay "información extra" en el texto.
5. **Generalización temporal.** El snapshot es de un único momento; señales
   como `recent_commits` o `last_commit_days` caducan rápido.

---

## 7. Aplicaciones de negocio

- **Triage de reclutamiento técnico.** Filtrar de una lista de candidatos las
  cuentas cuya producción pública es claramente `low_value`/`template` antes
  de un screening manual.
- **Due diligence de adquisiciones / inversión.** Evaluar la madurez de
  ingeniería de un proyecto open source al considerar adoptarlo o invertir en él.
- **Métricas de ecosistema.** Mapear la salud de ingeniería de un conjunto
  amplio de repos (p.ej. una organización, un programa de incubación).
- **Auditoría de portfolios estudiantiles.** Apoyar bootcamps o programas
  universitarios para identificar dónde están los proyectos que sí muestran
  prácticas profesionales.

En todos estos casos, el sistema debe usarse como **una señal entre varias**,
no como una sentencia.

---

## 8. Respuestas a las preguntas del Track A

### Q1 — ¿Qué señales separan `intern`/`junior`/`senior`/`lead`?

Mirando `output/tables/signals_by_label.csv` (que es la media de cada señal
por clase, n=480), la frontera de madurez se construye sobre **cuatro ejes**:

1. **Volumen y tracción social** — `stars`, `forks`, `watchers`,
   `engagement_ratio`. `lead` los domina; `intern` está en cero o cerca.
2. **Tamaño del equipo y revisión** — `contributors`, `merged_prs`. `lead` y
   `senior` son multi-contribuidor y con PRs merged; `junior`/`intern` son
   solo-dev típicamente.
3. **Disciplina de ingeniería** — `has_cicd`, `has_license`, `releases`,
   `readme_length`. CI/CD y releases son **la firma característica de senior+**;
   `junior` puede tener CI/CD pero raramente formaliza releases; `intern`
   normalmente no tiene ninguna de las dos.
4. **Madurez temporal** — `repo_age_days`, `is_mature`. `lead` tiende a ser
   maduro y activo; `intern` suele ser muy joven o muy poco activo.

El baseline de Regresión Logística (Q4) confirma cuantitativamente este orden:
sus features más informativas (|coef| medio) son `is_mature`, `has_cicd`,
`has_license`, `merged_prs`, `has_description`, `repo_age_days`, `has_wiki`,
`readme_length`. Es decir, **señales de proceso (CI/CD, licencia, revisión)
pesan más que las señales de popularidad (stars, forks)** para separar madurez.

### Q2 — ¿Cómo distinguir `template`/`low_value` de complejidad real?

`template` y `low_value` se diferencian del resto del dataset por **señales de
trabajo original sostenido**, no por popularidad:

- **Contribuidores ≈ 1**, **PRs merged ≈ 0**: no hay flujo de revisión.
- **README muy corto o ausente** y **CI/CD apagado**: no hay infraestructura.
- **`is_template == 1`** o `is_fork == 1` cuando es boilerplate replicado.
- **`engagement_ratio` muy bajo a pesar de la edad**: si lleva años publicado
  y nadie lo usa ni lo bifurca, es señal débil de utilidad.
- **`description_length` corta y sin `topics`**: tampoco hay esfuerzo de
  documentación de superficie.

Operativamente, una **regla simple** que ya recupera la mayoría es:

> Si `contributors ≤ 1` **y** `merged_prs == 0` **y** `has_cicd == 0` **y**
> `readme_length < 500`, marcar como candidato a `template`/`low_value` y
> mandarlo a revisión manual antes de evaluar madurez.

El gráfico `output/figures/03_low_value_vs_real.png` (boxplots de
`contributors`, `merged_prs`, `readme_length`, `has_cicd` separando esos dos
buckets) lo evidencia.

### Q3 — ¿Es útil para reclutadores / startups / screening?

**Sí, como señal complementaria; nunca como veredicto.** El valor concreto:

- **Reclutadores** pueden ahorrar tiempo cortando obvios `low_value` y
  `template` antes de revisar a mano.
- **Startups con poco bandwidth** pueden ordenar candidatos por madurez de
  proyectos públicos para priorizar entrevistas.
- **Equipos técnicos** pueden auditar el portfolio público de un colaborador
  externo / freelancer antes de un contrato.

**Limitaciones que cualquier consumidor del sistema debe asumir:**

- **El repo público no captura el trabajo privado.** Mucha gente buena escribe
  todo su código bajo NDA. Un perfil con repos pequeños puede esconder años de
  ingeniería seria que nunca se publican.
- **Popularidad ≠ calidad.** Un repo con muchas stars puede ser un meme; uno
  con pocas stars puede ser un compilador. El modelo usa señales agregadas y se
  puede equivocar en los extremos.
- **Métricas gamificables.** `commits`, `releases` o `readme_length` se pueden
  inflar artificialmente. Cualquier despliegue en producción debe defenderse
  contra esto (ver `recent_commits` saturado a 1000/mes en el dataset, que
  introduce sesgo).
- **Sesgo socioeconómico.** Mantener proyectos públicos requiere tiempo libre
  y conexión estable; sistemáticamente penalizar a quien no lo tiene es
  discriminatorio. El sistema **no** debe usarse como filtro automático en
  procesos de selección.
- **Sesgo temporal.** Las señales de actividad caducan. Un repo "dormido" puede
  ser código terminado y útil, no código abandonado.
- **Sesgo del LLM etiquetador.** El criterio de "senior" lo define el prompt de
  DeepSeek; otra empresa con otro estándar de senior obtendría otras etiquetas.

La recomendación honesta es presentar este sistema como un **scoring
auxiliar con explicación** — devolver la categoría junto a las señales que la
sostienen — y dejar la decisión final a una persona.

### Q4 — Sensibilidad metodológica: BERT vs Regresión Logística

Comparamos **DistilBERT fine-tuneado sobre `text_summary`** contra una
**Regresión Logística** entrenada sobre las **26 señales numéricas originales**
con `class_weight='balanced'` y `StandardScaler` (mismos splits, evaluación
sobre el mismo test set, cruce por `text_summary` para asegurar
correspondencia 1-a-1).

| escope          | DistilBERT | Logistic Regression | Δ |
|-----------------|-----------:|--------------------:|--:|
| accuracy global | 0.68       | **0.69**            | +0.01 |
| F1 weighted     | 0.59       | **0.72**            | +0.13 |
| F1 macro        | 0.29       | **0.36**            | +0.07 |
| F1 `senior`     | 0.17       | **0.62**            | **+0.45** |
| F1 `junior`     | **0.78**   | 0.69                | −0.09 |
| F1 `lead`       | 0.81       | **0.88**            | +0.07 |

**Conclusiones que se derivan:**

1. **El texto no aporta información extra sobre los números.** El `text_summary`
   es básicamente una verbalización de las señales, así que un modelo lineal
   sobre los números originales es suficiente para resolver el problema.
2. **DistilBERT con 336 ejemplos colapsa `senior`.** La Regresión Logística
   (que ve etiquetas balanceadas porque pesa las clases minoritarias) recupera
   senior con un F1 de **0.62** vs **0.17** del BERT. El problema entonces no
   es "qué modelo usar" sino **qué hacer con el desbalance**.
3. **Para esta tarea, la complejidad de BERT no se justifica.** El baseline
   lineal es más simple, más rápido, más auditable (coeficientes interpretables)
   y mejor en métricas agregadas.
4. **Las clases con 0–1 muestras (`intern`, `template`, `low_value`)** siguen
   teniendo F1 = 0 incluso con baseline + class weights. Sin **más etiquetas
   de esas clases** (ya sea con muestreo activo, oversampling, o etiquetado
   manual), ningún clasificador supervisado las va a aprender.

La conclusión metodológica del Track A es entonces: en problemas chicos y
desbalanceados donde el texto es derivado de features tabulares, **el orden
correcto es probar baselines numéricos primero, y reservar BERT (o cualquier
modelo grande) para cuando haya información textual genuinamente nueva**
(p.ej. descripciones humanas, READMEs largos sin resumir).

---

## 9. Estructura del repo

```
data/
  raw/                # snapshot crudo de la GitHub API
  processed/          # repos_processed.csv, repos_with_summary.csv
  labeled/            # repos_labeled.csv (480 + label + confidence + reasoning)
  splits/             # train.csv, val.csv, test.csv, test_predictions.csv
models/trained_models/  # checkpoint DistilBERT + label_classes.json
output/
  metrics/            # metrics.json (train.py) + evaluation_metrics.json + baseline_metrics.json
  tables/             # confusion_matrix, confused_pairs, error_examples,
                      # class_imbalance, signals_by_label, baseline_vs_bert,
                      # baseline_predictions, baseline_feature_importance
  figures/            # 6 PNGs (distribución, señales, low_value, matriz, baseline, confianza)
src/
  github_collector.py # Persona 1
  preprocessing.py    # Persona 1
  summarization.py    # Persona 1
  llm_labeling.py     # Persona 2
  train.py            # Persona 2
  evaluation.py       # Persona 3 — métricas + baseline + análisis de errores
  visualization.py    # Persona 3 — todas las figuras
app.py                # Persona 3 — dashboard de Streamlit (4 pestañas)
requirements.txt
```
