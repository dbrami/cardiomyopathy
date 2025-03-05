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

# --- Device Detection ---
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# --- Hyperparameters (adjust these as needed) ---
NUM_EPOCHS = 3             # Number of training epochs
LEARNING_RATE = 5e-5         # Learning rate for fine-tuning
TRAIN_BATCH_SIZE = 16        # Training batch size
EVAL_BATCH_SIZE = 16         # Evaluation batch size

# --------- Data Loading Functions ---------
def load_sirna_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["Sense sequence", "mRNA knockdown numeric"])
    return df

def load_promoters(fasta_path):
    promoters = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        # Extract the gene name (first token after '>')
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
        # For regression we need a float label.
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
    # Due to known issues with MPS and some embedding operations,
    # temporarily move the model and inputs to CPU for inference.
    model_cpu = model.to("cpu")
    encodings = tokenizer(candidates, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    encodings = {k: v.to("cpu") for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model_cpu(**encodings)
    predictions = outputs.logits.squeeze().tolist()
    if isinstance(predictions, float):
        predictions = [predictions]
    # Move the model back to the detected device.
    model.to(device)
    return predictions

# --------- Main Function ---------
def main():
    # Load the siRNA data
    sirna_df = load_sirna_data(SIRNA_DATA_PATH)
    sequences = sirna_df["Sense sequence"].tolist()
    labels = sirna_df["mRNA knockdown numeric"].tolist()

    # ---- Model and Tokenizer Setup using Evo2 from Hugging Face ----
    # Evo2_1b_base is a 1B-parameter model. Ensure you have enough memory.
    config = AutoConfig.from_pretrained("arcinstitute/evo2_1b_base")
    config.num_labels = 1  # Set number of labels for regression
    model = AutoModelForSequenceClassification.from_pretrained(
        "arcinstitute/evo2_1b_base",
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained("arcinstitute/evo2_1b_base")
    model.to(device)
    max_length = 50  # Maximum sequence length

    # Create the dataset and split into training and validation sets
    dataset = SiRNADataset(sequences, labels, tokenizer, max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Fine-tune the Evo2 model
    trainer = fine_tune_model(train_dataset, val_dataset, model, OUTPUT_DIR)

    # Load promoter regions
    promoters = load_promoters(PROMOTER_SEQ_PATH)

    # For each promoter, generate candidate probes and predict their efficiency.
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

    if candidate_results_list:
        final_results = pd.concat(candidate_results_list, ignore_index=True)
    else:
        final_results = pd.DataFrame(columns=["Gene", "Candidate", "Predicted_Efficiency"])

    candidates_path = os.path.join(OUTPUT_DIR, "predicted_sirna_candidates.csv")
    final_results.to_csv(candidates_path, index=False)
    print(f"Candidate predictions saved to {candidates_path}")

    # Generate visualizations and summary report (including epoch-based metric plots)
    visualizations.generate_report(final_results, trainer)

if __name__ == "__main__":
    main()