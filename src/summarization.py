import os
import pandas as pd

# ──────────────────────────────────────────
# GENERACIÓN DE RESÚMENES TEXTUALES
# Convierte señales numéricas en texto
# que el LLM y BERT puedan entender
# ──────────────────────────────────────────

def _i(row, key, default=0):
    """Lectura segura como entero (tolera NaN/strings)."""
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default

def _f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default

def _s(row, key):
    """Lectura segura como string limpio.
    Maneja NaN (float), None, y el string literal "nan" que aparece cuando
    pandas releee un CSV con celdas vacías.
    """
    v = row.get(key, "")
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return "" if s.lower() == "nan" else s

def describe_stars(stars):
    if stars == 0:
        return "no stars"
    elif stars <= 5:
        return f"{stars} stars (very low visibility)"
    elif stars <= 20:
        return f"{stars} stars (low visibility)"
    elif stars <= 100:
        return f"{stars} stars (moderate visibility)"
    elif stars <= 1000:
        return f"{stars} stars (good visibility)"
    else:
        return f"{stars} stars (high visibility)"

def describe_forks(forks):
    if forks == 0:
        return "no forks"
    elif forks <= 5:
        return f"{forks} forks (very few)"
    elif forks <= 50:
        return f"{forks} forks (modest community interest)"
    elif forks <= 500:
        return f"{forks} forks (notable community interest)"
    else:
        return f"{forks} forks (widely reused project)"

def describe_size(size_kb):
    if size_kb < 100:
        return "very small codebase"
    elif size_kb < 1000:
        return "small codebase"
    elif size_kb < 10000:
        return "medium-sized codebase"
    elif size_kb < 100000:
        return "large codebase"
    else:
        return "very large codebase"

def describe_issues(open_issues, repo_age_days):
    if open_issues == 0:
        return "no open issues"
    years = max(repo_age_days / 365.0, 0.1)
    rate = open_issues / years
    if rate < 1:
        return f"{open_issues} open issues (issue tracker rarely used)"
    elif rate < 10:
        return f"{open_issues} open issues (active issue tracking)"
    else:
        return f"{open_issues} open issues (heavy issue activity)"

def describe_commits(recent_commits, commit_rate):
    if recent_commits == 0:
        return "no commits in last 90 days"
    elif recent_commits <= 5:
        return f"{recent_commits} commits in last 90 days (~{commit_rate}/month, very low activity)"
    elif recent_commits <= 20:
        return f"{recent_commits} commits in last 90 days (~{commit_rate}/month, low activity)"
    elif recent_commits <= 60:
        return f"{recent_commits} commits in last 90 days (~{commit_rate}/month, moderate activity)"
    else:
        return f"{recent_commits} commits in last 90 days (~{commit_rate}/month, high activity)"

def describe_contributors(contributors):
    if contributors <= 1:
        return "solo project (1 contributor)"
    elif contributors <= 3:
        return f"{contributors} contributors (small team)"
    elif contributors <= 10:
        return f"{contributors} contributors (medium team)"
    else:
        return f"{contributors} contributors (large team)"

def describe_age(repo_age_days):
    if repo_age_days <= 7:
        return "brand new repository (less than 1 week old)"
    elif repo_age_days <= 30:
        return f"{repo_age_days} days old (very recent)"
    elif repo_age_days <= 180:
        return f"{repo_age_days} days old (recent)"
    elif repo_age_days <= 365:
        return f"{repo_age_days} days old (established)"
    else:
        years = round(repo_age_days / 365, 1)
        return f"{years} years old (mature repository)"

def describe_last_commit(last_commit_days):
    if last_commit_days <= 0:
        return "updated today"
    elif last_commit_days <= 7:
        return f"last commit {last_commit_days} days ago (very active)"
    elif last_commit_days <= 30:
        return f"last commit {last_commit_days} days ago (active)"
    elif last_commit_days <= 90:
        return f"last commit {last_commit_days} days ago (moderately active)"
    elif last_commit_days <= 365:
        return f"last commit {last_commit_days} days ago (low activity)"
    else:
        return f"last commit {last_commit_days} days ago (possibly abandoned)"

def describe_readme(readme_len):
    if readme_len == 0:
        return "no README"
    elif readme_len < 500:
        return f"minimal README ({readme_len} chars)"
    elif readme_len < 2000:
        return f"basic README ({readme_len} chars)"
    else:
        return f"detailed README ({readme_len} chars)"

def maturity_hint(row):
    """Sintetiza combinaciones de señales como lo haría un humano antes de etiquetar.
    NO emite la categoría — solo verbaliza el conjunto de señales para que el LLM
    tenga un resumen explícito que pueda usar como contexto para razonar.
    """
    cues = []

    is_template  = _i(row, "is_template")
    is_archived  = _i(row, "is_archived")
    is_fork      = _i(row, "is_fork")
    contribs     = _i(row, "contributors")
    cicd         = _i(row, "has_cicd")
    readme       = _i(row, "readme_length")
    releases     = _i(row, "releases")
    merged_prs   = _i(row, "merged_prs")
    last_commit  = _i(row, "last_commit_days")
    is_mature    = _i(row, "is_mature")
    is_active    = _i(row, "is_active")
    has_wiki     = _i(row, "has_wiki")
    has_pages    = _i(row, "has_pages")
    has_license  = _i(row, "has_license")
    topics_count = _i(row, "topics_count")
    desc_len     = _i(row, "description_length")

    if is_template:
        cues.append("explicitly marked as template")
    if is_archived:
        cues.append("archived (no longer maintained)")
    if is_fork:
        cues.append("fork of another project")

    # Equipo
    if contribs <= 1:
        cues.append("solo developer")
    elif contribs <= 3:
        cues.append("very small team")
    else:
        cues.append("multi-contributor team")

    # Automatización / workflow
    cues.append("automated CI/CD pipeline" if cicd else "no automated CI/CD")
    if merged_prs >= 50:
        cues.append("established PR-review workflow")
    elif merged_prs > 0:
        cues.append("some PR review activity")
    else:
        cues.append("no visible PR-review workflow")

    if releases >= 10:
        cues.append("regular release cadence")
    elif releases > 0:
        cues.append("a few releases published")
    else:
        cues.append("no formal releases")

    # Documentación
    if readme < 200 and desc_len < 30:
        cues.append("scarce documentation")
    elif readme > 3000:
        cues.append("extensive documentation")
    if has_wiki:
        cues.append("wiki enabled")
    if has_pages:
        cues.append("GitHub Pages site published")
    if has_license:
        cues.append("open-source license")
    if topics_count >= 3:
        cues.append("well-tagged with topics")
    elif topics_count == 0:
        cues.append("no topics declared")

    # Longevidad y actividad
    if is_mature and is_active:
        cues.append("mature and currently active")
    elif is_mature and not is_active:
        cues.append("mature but inactive")
    elif not is_mature and is_active:
        cues.append("young and actively developed")
    if last_commit > 365:
        cues.append("possibly abandoned")

    return "Combined engineering signals: " + ", ".join(cues) + "."

def generate_summary(row):
    """Genera el resumen textual de un repositorio.
    Orden pensado para que las señales más discriminativas aparezcan primero,
    en caso de que el LLM trunque el input.
    """
    parts = []

    # 1) Identidad y contexto temático
    desc = _s(row, "description")
    if desc:
        parts.append(f"Description: {desc}.")

    lang = _s(row, "language")
    if lang:
        parts.append(f"Primary language: {lang}.")

    topics = _s(row, "topics")
    if topics:
        parts.append(f"Topics: {topics}.")

    # 2) Perfil de ingeniería: tamaño y comunidad
    parts.append(describe_size(_i(row, "size_kb")) + ".")
    parts.append(describe_stars(_i(row, "stars")) + ".")
    parts.append(describe_forks(_i(row, "forks")) + ".")
    parts.append(describe_contributors(_i(row, "contributors")) + ".")

    # 3) Actividad y longevidad
    parts.append(
        describe_commits(_i(row, "recent_commits"), _f(row, "commit_rate")) + "."
    )
    parts.append(describe_last_commit(_i(row, "last_commit_days")) + ".")
    parts.append(describe_age(_i(row, "repo_age_days")) + ".")

    # 4) Workflow profesional
    merged_prs = _i(row, "merged_prs")
    releases = _i(row, "releases")
    open_issues = _i(row, "open_issues")
    age_days = _i(row, "repo_age_days")

    if merged_prs > 0:
        parts.append(f"{merged_prs} merged pull requests.")
    else:
        parts.append("No merged pull requests recorded.")
    if releases > 0:
        parts.append(f"{releases} releases published.")
    else:
        parts.append("No formal releases.")
    parts.append(describe_issues(open_issues, age_days) + ".")

    if _i(row, "has_cicd") == 1:
        parts.append("Has CI/CD workflows configured.")
    else:
        parts.append("No CI/CD workflows detected.")

    # 5) Documentación
    parts.append(describe_readme(_i(row, "readme_length")) + ".")
    if _i(row, "has_wiki") == 1:
        parts.append("Wiki enabled.")
    if _i(row, "has_pages") == 1:
        parts.append("GitHub Pages site published.")
    if _i(row, "has_license") == 1:
        parts.append("Has an open source license.")

    # 6) Flags especiales
    if _i(row, "is_template") == 1:
        parts.append("Marked as template repository.")
    if _i(row, "is_archived") == 1:
        parts.append("Repository is archived.")
    if _i(row, "is_fork") == 1:
        parts.append("This is a fork of another repository.")

    # 7) Métricas derivadas
    engagement = _f(row, "engagement_ratio")
    parts.append(f"Engagement ratio (stars+2*forks+watchers per day of age): {engagement}.")

    # 8) Hint de combinación de señales
    parts.append(maturity_hint(row))

    return " ".join(parts)

def add_text_summaries(df):
    """Aplica la generación de resúmenes a todo el dataframe"""
    df["text_summary"] = df.apply(generate_summary, axis=1)
    print(f"✅ Resúmenes generados para {len(df)} repositorios")
    return df

def save_with_summaries(df, path="data/processed/repos_with_summary.csv"):
    """Guarda el dataframe con la columna text_summary.

    Escribimos a un archivo distinto de repos_processed.csv para no sobreescribir
    el output de preprocessing y permitir trazabilidad / regeneración limpia.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"💾 Guardado en {path}")

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv("data/processed/repos_processed.csv", encoding='utf-8-sig')
    df = add_text_summaries(df)
    save_with_summaries(df)

    print(f"\n✅ Summarization completa")
    print("\nEjemplo de resumen generado:")
    print(df["text_summary"].iloc[0])