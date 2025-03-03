"""
Tests for DNABERT-2 training functionality
"""

import pytest
import torch
from pathlib import Path
import shutil
import tempfile
from src.dnabert_trainer import DNABERTTrainer, SiRNADataset

# Test data
TEST_SEQUENCES = [
    "ATGCTAGCTAGCTAG",
    "GCTAGCTAGCTAGCT",
    "TAGCTAGCTAGCTAT"
]

@pytest.fixture(scope="module")
def model_name():
    """Get model name for testing"""
    return "zhihan1996/DNABERT-2-117M"

@pytest.fixture(scope="function")
def trainer(temp_dir, model_name):
    """Create DNABERTTrainer instance for testing"""
    trainer = DNABERTTrainer(
        model_name=model_name,
        output_dir=temp_dir
    )
    trainer.setup()
    return trainer

@pytest.mark.slow
def test_trainer_setup(trainer):
    """Test model and tokenizer setup"""
    assert trainer.tokenizer is not None
    assert trainer.model is not None
    assert isinstance(trainer.model, torch.nn.Module)

@pytest.mark.slow
def test_prepare_training_data(trainer):
    """Test training data preparation"""
    train_loader, val_loader = trainer.prepare_training_data(
        TEST_SEQUENCES,
        val_split=0.2,
        batch_size=2
    )
    
    # Check data loaders
    assert train_loader is not None
    assert val_loader is not None
    
    # Check batch contents
    batch = next(iter(train_loader))
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert isinstance(batch['input_ids'], torch.Tensor)
    assert isinstance(batch['attention_mask'], torch.Tensor)
    
    # Check shapes
    assert len(batch['input_ids'].shape) == 2  # [batch_size, seq_len]
    assert batch['input_ids'].shape == batch['attention_mask'].shape

@pytest.mark.slow
def test_generate_sequences(trainer):
    """Test sequence generation"""
    num_sequences = 3
    sequences = trainer.generate_sequences(
        num_sequences=num_sequences,
        max_length=20
    )
    
    # Check number of sequences
    assert len(sequences) == num_sequences
    
    # Check sequence properties
    for seq in sequences:
        assert isinstance(seq, str)
        assert len(seq) > 0
        assert all(c in 'ATCG' for c in seq)

@pytest.mark.slow
def test_evaluate_sequences(trainer):
    """Test sequence evaluation"""
    evaluations = trainer.evaluate_sequences(TEST_SEQUENCES)
    
    # Check evaluation results
    assert len(evaluations) == len(TEST_SEQUENCES)
    assert 'sequence' in evaluations.columns
    assert 'model_score' in evaluations.columns
    
    # Check score properties
    scores = evaluations['model_score'].values
    assert all(0 <= score <= 1 for score in scores)

@pytest.mark.slow
def test_save_and_load(trainer, temp_dir):
    """Test model saving and loading"""
    # Save configuration
    trainer.save_config()
    
    # Load saved model
    loaded_trainer = DNABERTTrainer.load_trained(temp_dir)
    
    # Check loaded model
    assert loaded_trainer.tokenizer is not None
    assert loaded_trainer.model is not None
    
    # Test loaded model can generate sequences
    sequences = loaded_trainer.generate_sequences(num_sequences=1)
    assert len(sequences) == 1

@pytest.mark.parametrize("batch_size,max_length", [
    (2, 32),
    (4, 64),
    (8, 128)
])
def test_dataset_creation(trainer, batch_size, max_length):
    """Test dataset creation with different parameters"""
    dataset = SiRNADataset(
        TEST_SEQUENCES,
        trainer.tokenizer,
        max_length=max_length
    )
    
    # Test length
    assert len(dataset) == len(TEST_SEQUENCES)
    
    # Test batch creation
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    batch = next(iter(dataloader))
    assert batch['input_ids'].shape[0] == min(batch_size, len(TEST_SEQUENCES))
    assert batch['input_ids'].shape[1] == max_length

def test_invalid_inputs(model_name, temp_dir):
    """Test handling of invalid inputs"""
    trainer = DNABERTTrainer(model_name, temp_dir)
    
    # Test invalid sequences
    with pytest.raises(ValueError):
        trainer.prepare_training_data(["INVALID"])
    
    # Test empty sequences
    with pytest.raises(ValueError):
        trainer.prepare_training_data([])
    
    # Test None input
    with pytest.raises(ValueError):
        trainer.evaluate_sequences(None)

@pytest.mark.slow
def test_training_loop(trainer, temp_dir):
    """Test complete training loop"""
    # Prepare data
    train_loader, val_loader = trainer.prepare_training_data(
        TEST_SEQUENCES,
        val_split=0.2,
        batch_size=2
    )
    
    # Run training
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=1,
        learning_rate=2e-5,
        warmup_steps=1,
        save_steps=1,
        eval_steps=1
    )
    
    # Check saved files
    checkpoint_files = list(Path(temp_dir).glob("checkpoint-*"))
    assert len(checkpoint_files) > 0
    
    # Check model outputs
    test_sequence = TEST_SEQUENCES[0]
    with torch.no_grad():
        inputs = trainer.tokenizer(
            test_sequence,
            return_tensors='pt'
        ).to(trainer.device)
        outputs = trainer.model(**inputs)
        
    assert outputs.logits is not None
    assert outputs.logits.shape[1] == len(test_sequence) + 2  # +2 for special tokens

@pytest.mark.integration
def test_end_to_end(temp_dir, model_name):
    """Test end-to-end training and generation pipeline"""
    # Initialize trainer
    trainer = DNABERTTrainer(model_name, temp_dir)
    trainer.setup()
    
    # Train model
    train_loader, val_loader = trainer.prepare_training_data(TEST_SEQUENCES)
    trainer.train(train_loader, val_loader, num_epochs=1)
    
    # Generate sequences
    sequences = trainer.generate_sequences(num_sequences=2)
    assert len(sequences) == 2
    
    # Evaluate sequences
    evaluations = trainer.evaluate_sequences(sequences)
    assert len(evaluations) == 2