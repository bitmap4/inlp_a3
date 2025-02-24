import torch
import torch.nn as nn
import torch.optim as optim
from helper import load_brown_corpus, build_vocab, generate_training_data_cbow, get_negative_sampling_distribution
import random
import numpy as np

# Select device: use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Optionally, initialize weights using Xavier initialization:
        nn.init.xavier_uniform_(self.input_embeddings.weight)
        nn.init.xavier_uniform_(self.output_embeddings.weight)
    
    def forward(self, context_words, target_words, negative_words):
        # context_words: shape (batch_size, context_size)
        # target_words: shape (batch_size)
        # negative_words: shape (batch_size, num_negatives)
        # Get embeddings for context words and average them:
        context_embeds = self.input_embeddings(context_words)  # (batch_size, context_size, embedding_dim)
        context_embeds = torch.mean(context_embeds, dim=1)       # (batch_size, embedding_dim)
        
        # Get embeddings for target words:
        target_embeds = self.output_embeddings(target_words)    # (batch_size, embedding_dim)
        
        # Positive score: dot product between context and target embeddings
        pos_score = torch.sum(context_embeds * target_embeds, dim=1)  # (batch_size)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)          # (batch_size)
        
        # Negative sampling: get embeddings for negative examples
        neg_embeds = self.output_embeddings(negative_words)           # (batch_size, num_negatives, embedding_dim)
        # Compute dot product between each negative sample and the context vector
        neg_score = torch.bmm(neg_embeds, context_embeds.unsqueeze(2)).squeeze(2)  # (batch_size, num_negatives)
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)  # (batch_size)
        
        # Combine positive and negative loss; take negative to minimize
        loss = - (pos_loss + neg_loss)
        return loss.mean()

def train_cbow(embedding_dim=300, window_size=3, min_count=5, num_negatives=5, epochs=5, batch_size=128, lr=0.001):
    # Load the Brown Corpus and build vocabulary
    sentences = load_brown_corpus()
    word2idx, idx2word, vocab = build_vocab(sentences, min_count)
    # Generate training data for CBOW: each sample is (context, target)
    training_data = generate_training_data_cbow(sentences, word2idx, window_size)
    vocab_size = len(word2idx)
    
    model = CBOWModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Get the negative sampling distribution and convert to a torch tensor
    neg_sampling_prob = get_negative_sampling_distribution(vocab, word2idx)
    neg_sampling_prob = torch.tensor(neg_sampling_prob, dtype=torch.float, device=device)
    
    # Training loop
    for epoch in range(epochs):
        random.shuffle(training_data)
        losses = []
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i: i+batch_size]
            # Prepare context and target batches
            context_batch = torch.tensor([sample[0] for sample in batch], dtype=torch.long, device=device)
            target_batch = torch.tensor([sample[1] for sample in batch], dtype=torch.long, device=device)
            
            # Vectorized negative sampling: sample negatives for the entire batch at once
            negative_batch = torch.multinomial(neg_sampling_prob, num_negatives * len(batch), replacement=True)
            negative_batch = negative_batch.view(len(batch), num_negatives).to(device)
            
            optimizer.zero_grad()
            loss = model(context_batch, target_batch, negative_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")
    
    return model, word2idx, idx2word

if __name__ == "__main__":
    model, word2idx, idx2word = train_cbow()
    # Save the learned input embeddings (word vectors) along with vocabulary mappings
    torch.save({'embeddings': model.input_embeddings.weight.data.cpu(), 'word2idx': word2idx, 'idx2word': idx2word}, "cbow.pt")
    print("CBOW embeddings saved to cbow.pt")
