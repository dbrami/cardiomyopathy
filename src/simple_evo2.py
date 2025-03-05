import os
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import random_split
from Bio import SeqIO
from tqdm import tqdm

import visualizations

# --- File Paths ---
DATA_DIR = "../data"
RESULTS_DIR = "../results"
SIRNA_DATA_PATH = os.path.join(DATA_DIR, "mit_sirna/enriched_sirna_data.csv")
PROMOTER_SEQ_PATH = os.path.join(DATA_DIR, "promoters/promoter_sequences.fasta")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "evo2")

# --- Device Detection for MacBook M3 using MPS ---
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# --- Hyperparameters (adjust these as needed) ---
NUM_EPOCHS = 30            # Number of training epochs
LEARNING_RATE = 5e-5       # Learning rate for fine-tuning
TRAIN_BATCH_SIZE = 16      # Training batch size
EVAL_BATCH_SIZE = 16       # Evaluation batch size

# --- Model Names as Global Variables ---
MODEL_NAMES = ["arcinstitute/evo2_1b_base", "facebook/esm2_t12_35M_UR50D"]

# --------- Data Loading Functions ---------
def load_sirna_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["Sense sequence", "mRNA knockdown numeric"])
    return df

def load_promoters(fasta_path):
    promoters = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        gene_name = record.description.split()[0][1:]
        promoters[gene_name] = str(record.seq)
    return promoters

# --------- Dataset Class ---------
class SiRNADataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.encodings = tokenizer(sequences, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# --------- Fine-tuning Function ---------
def fine_tune_model(train_dataset, val_dataset, model, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = predictions.squeeze()
        mse = ((predictions - labels) ** 2).mean().item()
        return {"mse": mse}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer

# --------- Candidate Generation and Prediction ---------
def generate_candidate_probes(promoter_seq, window_size=21, step=1):
    candidates = []
    for i in range(0, len(promoter_seq) - window_size + 1, step):
        candidate = promoter_seq[i : i + window_size]
        candidates.append(candidate)
    return candidates

def predict_candidates(model, tokenizer, candidates, max_length):
    # Move model to CPU for inference to avoid potential MPS issues
    model_cpu = model.to("cpu")
    encodings = tokenizer(candidates, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    encodings = {k: v.to("cpu") for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model_cpu(**encodings)
    predictions = outputs.logits.squeeze().tolist()
    if isinstance(predictions, float):
        predictions = [predictions]
    # Move model back to original device (MPS if available)
    model.to(device)
    return predictions

# --------- Main Function ---------
def main():
    # Load the siRNA data
    sirna_df = load_sirna_data(SIRNA_DATA_PATH)
    sequences = sirna_df["Sense sequence"].tolist()
    labels = sirna_df["mRNA knockdown numeric"].tolist()

    # ---- Model and Tokenizer Setup ----
    model = None
    tokenizer = None
    for model_name in MODEL_NAMES:
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.num_labels = 1  # Set number of labels for regression
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"Successfully loaded model: {model_name}")
            break
        except ValueError as e:
            print(f"Error loading {model_name}: {e}")
    if model is None:
        raise RuntimeError("Failed to load any of the specified models.")

    # Move model to MPS device if available
    model.to(device)
    max_length = 50  # Maximum sequence length

    # Create dataset and split
    dataset = SiRNADataset(sequences, labels, tokenizer, max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Fine-tune the model
    trainer = fine_tune_model(train_dataset, val_dataset, model, OUTPUT_DIR)

    # Load promoter regions
    promoters = load_promoters(PROMOTER_SEQ_PATH)

    # Generate and predict candidate probes
    candidate_results_list = []
    for gene, seq in tqdm(promoters.items(), desc="Processing promoters"):
        candidates = generate_candidate_probes(seq, window_size=21, step=1)
        predictions = predict_candidates(model, tokenizer, candidates, max_length)
        candidate_df = pd.DataFrame({
            "Candidate": candidates,
            "Predicted_Efficiency": predictions
        })
        candidate_df["Gene"] = gene
        candidate_results_list.append(candidate_df)

    final_results = pd.concat(candidate_results_list, ignore_index=True) if candidate_results_list else pd.DataFrame(columns=["Gene", "Candidate", "Predicted_Efficiency"])
    candidates_path = os.path.join(OUTPUT_DIR, "predicted_sirna_candidates.csv")
    final_results.to_csv(candidates_path, index=False)
    print(f"Candidate predictions saved to {candidates_path}")

    # Generate visualizations
    visualizations.generate_report(final_results, trainer)

if __name__ == "__main__":
    main()