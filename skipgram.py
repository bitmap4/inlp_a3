import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from helper import load_brown_corpus, build_vocab, generate_training_data_skipgram, get_negative_sampling_distribution, sample_negative_examples
import random

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
    
    def forward(self, center_words, context_words, negative_words):
        # Get embeddings for center and context words
        center_embeds = self.center_embeddings(center_words)        # (batch_size, dim)
        context_embeds = self.context_embeddings(context_words)     # (batch_size, dim)
        
        # Positive score and loss
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)
        
        # Negative sampling
        neg_embeds = self.context_embeddings(negative_words)  # (batch_size, num_negatives, dim)
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze()  # (batch_size, num_negatives)
        neg_loss = torch.sum(torch.nn.functional.logsigmoid(-neg_score), dim=1)
        
        return - (pos_loss + neg_loss).mean()

def train_skipgram(embedding_dim=100, window_size=2, min_count=5, num_negatives=5, epochs=5, batch_size=512):
    # Load corpus and build vocabulary
    sentences = load_brown_corpus()
    word2idx, idx2word, vocab = build_vocab(sentences, min_count)
    training_data = generate_training_data_skipgram(sentences, word2idx, window_size)
    vocab_size = len(word2idx)

    # Convert training data to tensors
    center_words = torch.tensor([pair[0] for pair in training_data], dtype=torch.long)
    context_words = torch.tensor([pair[1] for pair in training_data], dtype=torch.long)
    
    # Precompute negative sampling distribution
    neg_sampling_dist = get_negative_sampling_distribution(vocab, word2idx)
    
    # Vectorized negative sampling
    negative_samples = torch.tensor([
        sample_negative_examples(neg_sampling_dist, num_negatives, pair[1])
        for pair in training_data
    ], dtype=torch.long)

    # Create DataLoader
    dataset = TensorDataset(center_words, context_words, negative_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Model, optimizer, move to device
    model = SkipGramModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.SparseAdam(model.parameters(), lr=0.002)

    for epoch in range(epochs):
        total_loss = 0
        for center_batch, context_batch, negative_batch in dataloader:
            center_batch, context_batch, negative_batch = center_batch.to(device), context_batch.to(device), negative_batch.to(device)
            
            optimizer.zero_grad()
            loss = model(center_batch, context_batch, negative_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(dataloader):.4f}")

    # Save embeddings
    torch.save({'embeddings': model.center_embeddings.weight.data.cpu(), 'word2idx': word2idx, 'idx2word': idx2word}, "skipgram.pt")
    print("Skip-gram embeddings saved to skipgram.pt")

if __name__ == "__main__":
    train_skipgram()
# Sample usage : python skipgram.py