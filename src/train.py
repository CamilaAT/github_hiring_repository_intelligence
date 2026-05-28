import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

# ──────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                       else "cuda" if torch.cuda.is_available() 
                       else "cpu")
print(f"🖥️ Usando dispositivo: {DEVICE}")

# ──────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────

class RepoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ──────────────────────────────────────────
# ENTRENAMIENTO
# ──────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, 
                       attention_mask=attention_mask, 
                       labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), all_preds, all_labels

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Iniciando fine-tuning de DistilBERT...")

    # Cargar datos
    df = pd.read_csv("data/labeled/repos_labeled.csv", encoding='utf-8-sig')
    print(f"📥 Cargados {len(df)} repositorios etiquetados")
    print(f"Distribución:\n{df['label'].value_counts()}")

    # Encodear etiquetas
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    print(f"\nClases: {list(le.classes_)}")

    # Split 70/15/15
    # Classes with < 5 samples are grouped into a single bucket for stratification
    # to avoid ValueError when rare classes end up with only 1 member in the temp set
    MIN_STRATIFY = 5
    counts = pd.Series(df["label_id"].values).value_counts()
    rare_ids = set(counts[counts < MIN_STRATIFY].index)
    if rare_ids:
        print(f"⚠️  Clases con menos de {MIN_STRATIFY} muestras agrupadas para stratify: "
              f"{[le.classes_[i] for i in rare_ids]}")

    def make_stratify(y):
        grouped = np.where(np.isin(y, list(rare_ids)), -1, y)
        _, ucounts = np.unique(grouped, return_counts=True)
        return grouped if ucounts.min() >= 2 else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["text_summary"].values, df["label_id"].values,
        test_size=0.30, random_state=RANDOM_SEED, stratify=make_stratify(df["label_id"].values)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50, random_state=RANDOM_SEED, stratify=make_stratify(y_temp)
    )

    print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Guardar splits
    os.makedirs("data/splits", exist_ok=True)
    train_df = pd.DataFrame({"text_summary": X_train, "label_id": y_train})
    val_df = pd.DataFrame({"text_summary": X_val, "label_id": y_val})
    test_df = pd.DataFrame({"text_summary": X_test, "label_id": y_test})
    train_df.to_csv("data/splits/train.csv", index=False, encoding='utf-8-sig')
    val_df.to_csv("data/splits/val.csv", index=False, encoding='utf-8-sig')
    test_df.to_csv("data/splits/test.csv", index=False, encoding='utf-8-sig')
    print("💾 Splits guardados en data/splits/")

    # Tokenizer y modelo
    print("\n📦 Cargando DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_)
    ).to(DEVICE)

    # Datasets y dataloaders
    train_dataset = RepoDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = RepoDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = RepoDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Optimizer y scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # Entrenamiento
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"\n📚 Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_preds, val_labels = evaluate(model, val_loader)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models/trained_models", exist_ok=True)
            model.save_pretrained("models/trained_models/distilbert_hiring")
            tokenizer.save_pretrained("models/trained_models/distilbert_hiring")
            print("✅ Mejor modelo guardado")

    # Evaluación final en test — recargar el mejor checkpoint, no el modelo del último epoch
    print("\n🧪 Evaluación final en test set...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "models/trained_models/distilbert_hiring"
    ).to(DEVICE)
    _, test_preds, test_labels = evaluate(model, test_loader)

    all_label_ids = list(range(len(le.classes_)))
    report = classification_report(
        test_labels, test_preds,
        labels=all_label_ids,
        target_names=le.classes_,
        zero_division=0,
        output_dict=True
    )
    print(classification_report(test_labels, test_preds,
                                labels=all_label_ids,
                                target_names=le.classes_,
                                zero_division=0))

    # Guardar predicciones y métricas
    os.makedirs("output/metrics", exist_ok=True)
    test_results = pd.DataFrame({
        "text_summary": X_test,
        "true_label": le.inverse_transform(test_labels),
        "predicted_label": le.inverse_transform(test_preds)
    })

    # Agregar repo_name al test set — dedup por text_summary para evitar rows duplicadas en el merge
    test_df_full = df[df["text_summary"].isin(X_test)][
        ["repo_name", "text_summary", "label"]
    ].drop_duplicates(subset="text_summary").copy()
    test_results = test_results.merge(
        test_df_full[["repo_name", "text_summary"]], 
        on="text_summary", how="left"
    )
    test_results.to_csv("data/splits/test_predictions.csv", 
                        index=False, encoding='utf-8-sig')

    with open("output/metrics/metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # Guardar label encoder
    with open("models/trained_models/label_classes.json", "w") as f:
        json.dump(list(le.classes_), f)

    print("\n✅ Entrenamiento completo")
    print(f"📁 Predicciones: data/splits/test_predictions.csv")
    print(f"📁 Métricas: output/metrics/metrics.json")