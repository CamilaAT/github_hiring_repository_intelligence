import os
import time
import json
import pandas as pd
from github import Github, Auth
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
g = Github(auth=Auth.Token(GITHUB_TOKEN))


def _core_rate():
    """Devuelve el objeto de rate limit core, compatible con PyGithub 1.x y 2.x."""
    rl = g.get_rate_limit()
    # PyGithub 2.x: RateLimitOverview -> .resources.core
    if hasattr(rl, "resources"):
        return rl.resources.core
    # PyGithub 1.x: RateLimit -> .core
    return rl.core

# ──────────────────────────────────────────
# SEÑALES A EXTRAER POR REPOSITORIO
# ──────────────────────────────────────────

def get_contributor_count(repo):
    """Cuenta contribuidores usando totalCount (1 request)."""
    try:
        return repo.get_contributors().totalCount
    except Exception:
        return 0

def get_recent_commits(repo, days=90):
    """Cuenta commits en los últimos `days` días."""
    try:
        from datetime import datetime, timedelta, timezone
        since = datetime.now(timezone.utc) - timedelta(days=days)
        commits = repo.get_commits(since=since)
        count = 0
        for _ in commits:
            count += 1
            if count >= 1000:
                break
        return count
    except Exception:
        return 0

def get_merged_prs(repo, max_scan=300):
    """Cuenta PRs mergeados escaneando como máximo `max_scan` PRs cerrados."""
    try:
        prs = repo.get_pulls(state='closed')
        merged = 0
        scanned = 0
        for pr in prs:
            scanned += 1
            if pr.merged_at:
                merged += 1
            if scanned >= max_scan:
                break
        return merged
    except Exception:
        return 0

def has_cicd(repo):
    """Detecta si tiene workflows de CI/CD"""
    try:
        contents = repo.get_contents(".github/workflows")
        return 1 if contents else 0
    except:
        return 0

def get_readme_length(repo):
    """Retorna la longitud del README en caracteres"""
    try:
        readme = repo.get_readme()
        return len(readme.decoded_content.decode('utf-8', errors='ignore'))
    except:
        return 0

def extract_repo_signals(repo):
    """Extrae todas las señales de un repositorio"""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    created_at = repo.created_at.replace(tzinfo=timezone.utc) if repo.created_at else now
    pushed_at = repo.pushed_at.replace(tzinfo=timezone.utc) if repo.pushed_at else now

    repo_age_days = (now - created_at).days
    last_commit_days = (now - pushed_at).days

    signals = {
        "repo_id":           repo.id,
        "repo_name":         repo.full_name,
        "description":       repo.description or "",
        "stars":             repo.stargazers_count,
        "forks":             repo.forks_count,
        "watchers":          repo.watchers_count,
        "open_issues":       repo.open_issues_count,
        "size_kb":           repo.size,
        "language":          repo.language,
        "topics":            ", ".join(repo.get_topics()),
        "repo_age_days":     repo_age_days,
        "last_commit_days":  last_commit_days,
        "releases":          repo.get_releases().totalCount,
        "contributors":      get_contributor_count(repo),
        "recent_commits":    get_recent_commits(repo),
        "merged_prs":        get_merged_prs(repo),
        "has_cicd":          has_cicd(repo),
        "readme_length":     get_readme_length(repo),
        "is_fork":           int(repo.fork),
        "is_template":       int(bool(getattr(repo, "is_template", False))),
        "is_archived":       int(bool(getattr(repo, "archived", False))),
        "has_license":       int(repo.license is not None) if hasattr(repo, "license") else 0,
        "has_wiki":          int(bool(getattr(repo, "has_wiki", False))),
        "has_pages":         int(bool(getattr(repo, "has_pages", False))),
        "has_description":   int(repo.description is not None and len(repo.description) > 0),
    }
    return signals

# ──────────────────────────────────────────
# ESTRATEGIA DE SAMPLEO
# Buscamos repos en distintos rangos de stars
# para capturar todos los niveles de madurez
# ──────────────────────────────────────────

SEARCH_QUERIES = [
    # Nivel bajo (posibles intern/template/low-value)
    ("stars:0..5 size:>10 is:public fork:false", 80),
    ("stars:6..20 size:>10 is:public fork:false", 80),
    # Nivel medio (posibles junior/senior)
    ("stars:21..100 size:>50 is:public fork:false", 100),
    ("stars:101..500 size:>50 is:public fork:false", 100),
    # Nivel alto (posibles senior/lead)
    ("stars:501..2000 size:>100 is:public fork:false", 120),
    ("stars:>2000 size:>100 is:public fork:false", 120),
]

# ──────────────────────────────────────────
# RECOLECCIÓN PRINCIPAL
# ──────────────────────────────────────────

def check_rate_limit(min_remaining=200):
    """Si quedan pocos requests, espera hasta el reset."""
    from datetime import datetime, timezone
    try:
        rl = _core_rate()
        if rl.remaining < min_remaining:
            reset_utc = rl.reset.replace(tzinfo=timezone.utc)
            reset_in = (reset_utc - datetime.now(timezone.utc)).total_seconds()
            wait = max(reset_in, 0) + 5
            print(f"⏸️  Rate limit bajo ({rl.remaining}). Esperando {int(wait)}s hasta reset...")
            time.sleep(wait)
    except Exception:
        pass


def collect_repositories(target=400):
    all_repos = []
    seen_ids = set()

    for query, limit in SEARCH_QUERIES:
        print(f"\n🔍 Buscando: {query} (límite: {limit})")
        check_rate_limit()
        try:
            results = g.search_repositories(query=query, sort="updated")
            count = 0
            for repo in tqdm(results, total=limit):
                if count >= limit:
                    break
                if repo.id in seen_ids:
                    continue
                try:
                    signals = extract_repo_signals(repo)
                    all_repos.append(signals)
                    seen_ids.add(repo.id)
                    count += 1

                    # Guardar progreso cada 25 repos
                    if len(all_repos) % 25 == 0:
                        save_progress(all_repos)
                        print(f"💾 Progreso guardado: {len(all_repos)} repos")
                        check_rate_limit()

                    # Respetar rate limit secundario
                    time.sleep(0.5)

                except Exception as e:
                    print(f"⚠️ Error en {repo.full_name}: {e}")
                    continue

        except Exception as e:
            print(f"❌ Error en query '{query}': {e}")
            time.sleep(10)
            continue

        if len(all_repos) >= target:
            break

    return all_repos

def save_progress(repos):
    """Guarda los datos recolectados hasta el momento (UTF-8)."""
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/repos_raw.json", "w", encoding="utf-8") as f:
        json.dump(repos, f, indent=2, default=str, ensure_ascii=False)
    # utf-8-sig añade BOM para que Excel abra el CSV con tildes/emojis correctamente
    pd.DataFrame(repos).to_csv("data/raw/repos_raw.csv", index=False, encoding="utf-8-sig")

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Iniciando recolección de repositorios GitHub...")
    print(f"Rate limit disponible: {_core_rate().remaining} requests")

    repos = collect_repositories(target=400)
    save_progress(repos)

    print(f"\n✅ Recolección completa: {len(repos)} repositorios")
    print(f"📁 Guardado en data/raw/repos_raw.csv")
    print(f"Rate limit restante: {_core_rate().remaining} requests")