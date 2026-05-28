# github_hiring_repository_intelligence

Weak-supervision NLP pipeline that classifies GitHub repositories by engineering maturity level (intern to lead) using GitHub API signals, LLM weak labeling, and a fine-tuned BERT model.

## Setup

1. **Clone & install dependencies**

   ```bash
   git clone <repo-url>
   cd github_hiring_repository_intelligence
   pip install -r requirements.txt
   ```

2. **GitHub Personal Access Token**

   The data collector calls the GitHub REST API and requires a Personal Access Token (PAT).

   - Create one at <https://github.com/settings/tokens>
   - Required scope: `public_repo` (read-only is enough)
   - Copy the example env file and paste your token:

     ```bash
     cp .env.example .env
     ```

     Then edit `.env` and replace the placeholder:

     ```
     GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     ```

   ⚠️ `.env` is gitignored — never commit your real token.

## Pipeline (Persona 1 — data layer)

```bash
python src/github_collector.py   # → data/raw/repos_raw.csv
python src/preprocessing.py      # → data/processed/repos_processed.csv
python src/summarization.py      # → data/processed/repos_with_summary.csv
```

The final `repos_with_summary.csv` (column `text_summary`) is the input for the LLM weak-labeling stage handled by Persona 2.
