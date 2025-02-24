import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from helper import load_brown_corpus, build_vocab, generate_training_data_skipgram, get_negative_sampling_distribution, sample_negative_examples
from tqdm import tqdm
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        
        # Improved initialization
        with torch.no_grad():
            nn.init.xavier_uniform_(self.center_embeddings.weight)
            nn.init.xavier_uniform_(self.context_embeddings.weight)
    
    def forward(self, center_words, context_words, negative_words):
        # Get embeddings with better memory management
        center_embeds = self.center_embeddings(center_words)
        context_embeds = self.context_embeddings(context_words)
        
        # Compute positive scores efficiently
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)
        
        # Optimized negative sampling computation
        neg_embeds = self.context_embeddings(negative_words)
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(torch.nn.functional.logsigmoid(-neg_score), dim=1)
        
        return -(pos_loss + neg_loss).mean()

def train_skipgram(
    embedding_dim=100,
    window_size=2,
    min_count=5,
    num_negatives=5,
    epochs=5,
    batch_size=512,
    learning_rate=0.002,
    num_workers=4
):
    """
    Train Skip-gram model with improved efficiency and monitoring.
    
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
        training_data = generate_training_data_skipgram(sentences, word2idx, window_size)
        
        # Efficient tensor conversion
        center_words = torch.tensor([pair[0] for pair in training_data], dtype=torch.long)
        context_words = torch.tensor([pair[1] for pair in training_data], dtype=torch.long)
        
        logger.info("Computing negative sampling distribution...")
        neg_sampling_dist = get_negative_sampling_distribution(vocab, word2idx)
        
        logger.info("Generating negative samples...")
        negative_samples = torch.tensor([
            sample_negative_examples(neg_sampling_dist, num_negatives, pair[1])
            for pair in tqdm(training_data, desc="Generating negative samples")
        ], dtype=torch.long)

        # Create efficient DataLoader
        dataset = TensorDataset(center_words, context_words, negative_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Initialize model and optimizer
        model = SkipGramModel(vocab_size, embedding_dim).to(device)
        optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate)
        
        # Training loop with progress bar
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for center_batch, context_batch, negative_batch in progress_bar:
                # Move batch to device efficiently
                center_batch = center_batch.to(device, non_blocking=True)
                context_batch = context_batch.to(device, non_blocking=True)
                negative_batch = negative_batch.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                loss = model(center_batch, context_batch, negative_batch)
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
                'embeddings': model.center_embeddings.weight.data.cpu(),
                'word2idx': word2idx,
                'idx2word': idx2word,
                'config': {
                    'embedding_dim': embedding_dim,
                    'window_size': window_size,
                    'min_count': min_count
                }
            },
            "skipgram.pt"
        )
        logger.info("Model saved successfully to skipgram.pt")
        
        return model, word2idx, idx2word
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_skipgram()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")