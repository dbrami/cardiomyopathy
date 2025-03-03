"""
DNABERT-2 Fine-tuning Module

This module implements functionality for fine-tuning DNABERT-2 on siRNA sequences
and generating new candidate sequences.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class SiRNADataset(Dataset):
    """Custom dataset for siRNA sequences"""
    
    def __init__(self, sequences: List[str], tokenizer, max_length: int = 128):
        """
        Initialize dataset
        
        Args:
            sequences: List of DNA sequences
            tokenizer: DNABERT tokenizer
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Tokenize sequence
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class DNABERTTrainer:
    """Handles DNABERT-2 model fine-tuning and sequence generation"""
    
    def __init__(self, 
                 model_name: str = "zhihan1996/DNABERT-2-117M",
                 output_dir: str = "models/dnabert2_finetuned",
                 device: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            model_name: Name/path of pretrained model
            output_dir: Directory for saving model artifacts
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def setup(self):
        """Setup tokenizer and model"""
        try:
            self.logger.info(f"Loading tokenizer and model from {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name,
                trust_remote_code=True
            ).to(self.device)
            
            self.logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error in setup: {e}")
            raise
            
    def prepare_training_data(self, 
                            sequences: List[str],
                            val_split: float = 0.1,
                            max_length: int = 128,
                            batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders
        
        Args:
            sequences: List of training sequences
            val_split: Validation set fraction
            max_length: Maximum sequence length
            batch_size: Batch size for training
            
        Returns:
            train_loader, val_loader: Training and validation data loaders
        """
        try:
            # Split data
            train_seqs, val_seqs = train_test_split(
                sequences,
                test_size=val_split,
                random_state=42
            )
            
            # Create datasets
            train_dataset = SiRNADataset(
                train_seqs,
                self.tokenizer,
                max_length
            )
            val_dataset = SiRNADataset(
                val_seqs,
                self.tokenizer,
                max_length
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            self.logger.info(f"Prepared {len(train_seqs)} training and {len(val_seqs)} validation sequences")
            
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise
            
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 10,
              learning_rate: float = 2e-5,
              warmup_steps: int = 500,
              save_steps: int = 1000,
              eval_steps: int = 500) -> None:
        """
        Fine-tune the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            save_steps: Steps between checkpoints
            eval_steps: Steps between evaluations
        """
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=train_loader.batch_size,
                per_device_eval_batch_size=val_loader.batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                weight_decay=0.01,
                logging_dir=str(self.output_dir / 'logs'),
                logging_steps=100,
                save_steps=save_steps,
                eval_steps=eval_steps,
                evaluation_strategy="steps",
                load_best_model_at_end=True,
                save_total_limit=3,
                fp16=torch.cuda.is_available(),
            )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_loader.dataset,
                eval_dataset=val_loader.dataset,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            # Train model
            self.logger.info("Starting model training")
            train_result = self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Save training metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
            self.logger.info("Training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
            
    def generate_sequences(self,
                         num_sequences: int = 10,
                         max_length: int = 128,
                         temperature: float = 1.0) -> List[str]:
        """
        Generate new siRNA sequences
        
        Args:
            num_sequences: Number of sequences to generate
            max_length: Maximum sequence length
            temperature: Sampling temperature
            
        Returns:
            List of generated sequences
        """
        try:
            self.model.eval()
            generated_sequences = []
            
            for _ in range(num_sequences):
                # Start with CLS token
                current_ids = torch.tensor([[self.tokenizer.cls_token_id]]).to(self.device)
                
                # Generate sequence auto-regressively
                with torch.no_grad():
                    for _ in range(max_length - 1):
                        outputs = self.model(current_ids)
                        next_token_logits = outputs.logits[:, -1, :] / temperature
                        next_token = torch.multinomial(
                            torch.softmax(next_token_logits, dim=-1),
                            num_samples=1
                        )
                        current_ids = torch.cat([current_ids, next_token], dim=1)
                        
                        # Stop if EOS token is generated
                        if next_token.item() == self.tokenizer.sep_token_id:
                            break
                
                # Decode sequence
                sequence = self.tokenizer.decode(
                    current_ids[0],
                    skip_special_tokens=True
                )
                generated_sequences.append(sequence)
            
            self.logger.info(f"Generated {len(generated_sequences)} sequences")
            return generated_sequences
            
        except Exception as e:
            self.logger.error(f"Error generating sequences: {e}")
            raise
            
    def evaluate_sequences(self, sequences: List[str]) -> pd.DataFrame:
        """
        Evaluate quality of generated sequences
        
        Args:
            sequences: List of sequences to evaluate
            
        Returns:
            DataFrame with sequence evaluations
        """
        try:
            results = []
            
            for sequence in sequences:
                # Get model predictions
                inputs = self.tokenizer(
                    sequence,
                    return_tensors='pt',
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Calculate sequence score
                logits = outputs.logits.mean(dim=1).squeeze()
                score = torch.softmax(logits, dim=0).max().item()
                
                results.append({
                    'sequence': sequence,
                    'length': len(sequence),
                    'model_score': score
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Error evaluating sequences: {e}")
            raise
            
    def save_config(self):
        """Save training configuration"""
        config = {
            'model_name': self.model_name,
            'device': self.device,
            'tokenizer_config': self.tokenizer.config.to_dict(),
            'model_config': self.model.config.to_dict()
        }
        
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load_trained(cls, model_dir: str) -> 'DNABERTTrainer':
        """
        Load trained model from directory
        
        Args:
            model_dir: Directory containing saved model
            
        Returns:
            Initialized trainer with loaded model
        """
        try:
            model_dir = Path(model_dir)
            
            # Load config
            with open(model_dir / 'config.json', 'r') as f:
                config = json.load(f)
            
            # Initialize trainer
            trainer = cls(
                model_name=str(model_dir),
                output_dir=str(model_dir),
                device=config['device']
            )
            
            # Load tokenizer and model
            trainer.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            trainer.model = AutoModelForMaskedLM.from_pretrained(str(model_dir))
            trainer.model.to(trainer.device)
            
            return trainer
            
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            raise