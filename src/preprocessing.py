import os
import pandas as pd
import numpy as np

# Ventana usada por el collector para "recent_commits".
# Si cambias esto en github_collector.get_recent_commits, cámbialo aquí también.
WINDOW_DAYS = 90

# ──────────────────────────────────────────
# LIMPIEZA Y NORMALIZACIÓN
# ──────────────────────────────────────────

def load_raw_data(path="data/raw/repos_raw.csv"):
    """Carga el CSV crudo"""
    df = pd.read_csv(path, encoding='utf-8-sig')
    print(f"📥 Cargados {len(df)} repositorios")
    return df

def remove_duplicates(df):
    """Elimina repositorios duplicados"""
    before = len(df)
    df = df.drop_duplicates(subset="repo_id")
    print(f"🔁 Duplicados eliminados: {before - len(df)}")
    return df

def handle_missing_values(df):
    """Rellena valores nulos"""
    # Columnas numéricas → 0
    numeric_cols = [
        "stars", "forks", "watchers", "open_issues", "size_kb",
        "repo_age_days", "last_commit_days", "releases", "contributors",
        "recent_commits", "merged_prs", "has_cicd", "readme_length",
        "is_fork", "is_template", "is_archived", "has_license",
        "has_wiki", "has_pages", "has_description"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Columnas de texto → string vacío
    text_cols = ["repo_name", "description", "language", "topics"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    print("✅ Valores nulos manejados")
    return df

def drop_extraction_failures(df):
    """Elimina SOLO filas con fallas de extracción (no decisiones de calidad).

    Justificación metodológica: en Track A, los repos vacíos / sin actividad
    son ejemplos válidos de la categoría `low_value` y `template`. Eliminarlos
    sesgaría el dataset y privaría al modelo de los training examples que
    justamente debe aprender a detectar. Aquí solo descartamos filas donde el
    collector no logró obtener identificadores básicos (errores técnicos,
    no juicios sobre el contenido del repo).
    """
    before = len(df)
    df = df[df["repo_name"].astype(str).str.strip() != ""]
    df = df[df["repo_id"].notna()]
    dropped = before - len(df)
    if dropped > 0:
        print(f"🗑️  Filas con fallas de extracción descartadas: {dropped}")
    else:
        print("✅ Sin fallas de extracción — se conservan todos los repos")
    return df

def add_derived_features(df):
    """Agrega features derivados útiles para clasificación y para el resumen."""
    # Actividad reciente normalizada a commits/mes
    df["commit_rate"] = (df["recent_commits"] / (WINDOW_DAYS / 30)).round(2)

    # Engagement: stars + forks (peso doble por costar más interés) + watchers,
    # normalizado por edad para no premiar repos viejos por inercia.
    df["engagement_ratio"] = (
        (df["stars"] + df["forks"] * 2 + df["watchers"]) /
        (df["repo_age_days"] + 1)
    ).round(4)

    # Repo maduro (>6 meses de existencia)
    df["is_mature"] = (df["repo_age_days"] > 180).astype(int)

    # Actividad reciente (commit en últimos 30 días)
    df["is_active"] = (df["last_commit_days"] <= 30).astype(int)

    # Largo de descripción y cantidad de topics — útiles tanto para el LLM
    # como para visualizaciones (los repos serios tienden a documentar y taggear).
    df["description_length"] = df["description"].astype(str).str.len()
    df["topics_count"] = df["topics"].astype(str).apply(
        lambda s: 0 if not s.strip() else len([t for t in s.split(",") if t.strip()])
    )

    print("➕ Features derivados agregados")
    return df

def save_processed(df, path="data/processed/repos_processed.csv"):
    """Guarda el dataset procesado"""
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"💾 Guardado en {path} ({len(df)} repos)")

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    df = load_raw_data()
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = drop_extraction_failures(df)
    df = add_derived_features(df)
    save_processed(df)
    print(f"\n✅ Preprocesamiento completo: {len(df)} repos listos")
    print(df.head())