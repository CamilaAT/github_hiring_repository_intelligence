import os
import time
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ──────────────────────────────────────────
# CATEGORÍAS Y PROMPT
# ──────────────────────────────────────────

CATEGORIES = ["intern", "junior", "senior", "lead", "template", "low_value"]

SYSTEM_PROMPT = """You are an expert software engineering recruiter and technical evaluator.
Your task is to classify GitHub repositories based on their engineering maturity level.

You must classify each repository into EXACTLY one of these categories:
- intern: Simple, learning-oriented projects. Basic code, minimal structure, no CI/CD, solo developer, very few commits.
- junior: Functional projects with some structure. Basic documentation, limited testing, small team or solo, moderate activity.
- senior: Well-structured projects with good practices. CI/CD present, multiple contributors, regular releases, good documentation.
- lead: Highly complex, mature projects. Large teams, extensive PR history, multiple releases, comprehensive docs, established ecosystem.
- template: Boilerplate, starter kits, or replica repositories with little original engineering work.
- low_value: Abandoned, empty, spam, or repositories with no meaningful engineering content.

Respond with ONLY a JSON object in this exact format:
{"label": "category_name", "confidence": 0.0, "reasoning": "brief explanation"}

Where confidence is a float between 0.0 and 1.0.
Do not include any other text."""

def classify_repo(text_summary):
    """Clasifica un repositorio usando DeepSeek"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this repository:\n\n{text_summary}"}
            ],
            max_tokens=150,
            temperature=0.1
        )
        
        raw = response.choices[0].message.content.strip()
        # Extract JSON object even if the model wraps it in markdown fences
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
        result = json.loads(raw)

        # Validar que la etiqueta sea válida
        label = result.get("label", "low_value")
        if label not in CATEGORIES:
            label = "low_value"
        result["label"] = label
            
        return result
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return {"label": "low_value", "confidence": 0.0, "reasoning": "API error"}

# ──────────────────────────────────────────
# ETIQUETADO PRINCIPAL
# ──────────────────────────────────────────

def label_repositories(df, save_every=25):
    """Etiqueta todos los repositorios con DeepSeek"""
    os.makedirs("data/labeled", exist_ok=True)

    labels = []
    confidences = []
    reasonings = []

    for counter, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Etiquetando repos")):
        result = classify_repo(row["text_summary"])

        labels.append(result["label"])
        confidences.append(result.get("confidence", 0.0))
        reasonings.append(result.get("reasoning", ""))

        # Guardar progreso cada N repos
        if (counter + 1) % save_every == 0:
            temp_df = df.iloc[:counter + 1].copy()
            temp_df["label"] = labels
            temp_df["confidence"] = confidences
            temp_df["reasoning"] = reasonings
            temp_df.to_csv("data/labeled/repos_labeled_partial.csv",
                          index=False, encoding='utf-8-sig')
            print(f"💾 Progreso guardado: {counter + 1} repos etiquetados")

        # Respetar rate limit de DeepSeek
        time.sleep(0.3)
    
    df["label"] = labels
    df["confidence"] = confidences
    df["reasoning"] = reasonings
    
    return df

def save_labeled(df, path="data/labeled/repos_labeled.csv"):
    """Guarda el dataset etiquetado"""
    os.makedirs("data/labeled", exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"💾 Guardado en {path} ({len(df)} repos)")

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Iniciando etiquetado con DeepSeek...")
    
    df = pd.read_csv("data/processed/repos_with_summary.csv", encoding='utf-8-sig')
    print(f"📥 Cargados {len(df)} repositorios")
    print(f"Columnas disponibles: {list(df.columns)}")
    
    # Test con 1 repo primero
    print("\n🧪 Test con 1 repositorio...")
    test_result = classify_repo(df["text_summary"].iloc[0])
    print(f"Resultado test: {test_result}")
    
    confirm = input("\n¿Todo OK? ¿Continuar con todos los repos? (s/n): ")
    if confirm.lower() != 's':
        print("Cancelado.")
        exit()
    
    df = label_repositories(df)
    save_labeled(df)
    
    print(f"\n✅ Etiquetado completo")
    print(f"Distribución de etiquetas:")
    print(df["label"].value_counts())