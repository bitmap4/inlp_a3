import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from helper import load_brown_corpus, build_vocab, generate_training_data_cbow, get_negative_sampling_distribution
from tqdm import tqdm
import logging
import random
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        
        # Improved initialization
        with torch.no_grad():
            nn.init.xavier_uniform_(self.input_embeddings.weight)
            nn.init.xavier_uniform_(self.output_embeddings.weight)
    
    def forward(self, context_words, target_words, negative_words):
        # Efficient computation of context embeddings
        context_embeds = self.input_embeddings(context_words)
        context_embeds = torch.mean(context_embeds, dim=1)
        
        # Compute positive scores efficiently
        target_embeds = self.output_embeddings(target_words)
        pos_score = torch.sum(context_embeds * target_embeds, dim=1)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)
        
        # Optimized negative sampling computation
        neg_embeds = self.output_embeddings(negative_words)
        neg_score = torch.bmm(neg_embeds, context_embeds.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(torch.nn.functional.logsigmoid(-neg_score), dim=1)
        
        return -(pos_loss + neg_loss).mean()

def train_cbow(
    embedding_dim=300,
    window_size=3,
    min_count=5,
    num_negatives=5,
    epochs=5,
    batch_size=512,
    learning_rate=0.001,
    num_workers=4
):
    """
    Train CBOW model with improved efficiency and monitoring.
    
    Args:
        embedding_dim: Dimension of word embeddings
        window_size: Context window size
        min_count: Minimum word frequency
        num_negatives: Number of negative samples
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        num_workers: Number of DataLoader workers
    """
    try:
        logger.info("Loading corpus and building vocabulary...")
        sentences = load_brown_corpus()
        word2idx, idx2word, vocab = build_vocab(sentences, min_count)
        vocab_size = len(word2idx)
        
        logger.info("Generating training data...")
        training_data = generate_training_data_cbow(sentences, word2idx, window_size)
        
        # Efficient tensor conversion
        context_words = torch.tensor([sample[0] for sample in training_data], dtype=torch.long)
        target_words = torch.tensor([sample[1] for sample in training_data], dtype=torch.long)
        
        logger.info("Computing negative sampling distribution...")
        neg_sampling_prob = torch.tensor(
            get_negative_sampling_distribution(vocab, word2idx),
            dtype=torch.float
        )
        
        # Generate negative samples efficiently
        logger.info("Generating negative samples...")
        negative_samples = torch.stack([
            torch.multinomial(neg_sampling_prob, num_negatives, replacement=True)
            for _ in tqdm(range(len(training_data)), desc="Generating negative samples")
        ])

        # Create efficient DataLoader
        dataset = TensorDataset(context_words, target_words, negative_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Initialize model and optimizer
        model = CBOWModel(vocab_size, embedding_dim).to(device)
        optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate)
        
        # Training loop with progress bar
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for context_batch, target_batch, negative_batch in progress_bar:
                # Move batch to device efficiently
                context_batch = context_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                negative_batch = negative_batch.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                loss = model(context_batch, target_batch, negative_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # Save model efficiently
        logger.info("Saving model...")
        torch.save(
            {
                'embeddings': model.input_embeddings.weight.data.cpu(),
                'word2idx': word2idx,
                'idx2word': idx2word,
                'config': {
                    'embedding_dim': embedding_dim,
                    'window_size': window_size,
                    'min_count': min_count
                }
            },
            "cbow.pt"
        )
        logger.info("Model saved successfully to cbow.pt")
        
        return model, word2idx, idx2word
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_cbow()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")