# skipgram.py
import torch
import torch.nn as nn
import torch.optim as optim
from helper import load_brown_corpus, build_vocab, generate_training_data_skipgram, get_negative_sampling_distribution, sample_negative_examples
import random
import numpy as np
# from tqdm import tqdm

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, center_words, context_words, negative_words):
        # Get embeddings for center and context words
        center_embeds = self.center_embeddings(center_words)      # (batch_size, dim)
        context_embeds = self.context_embeddings(context_words)     # (batch_size, dim)
        
        # Positive score and loss
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)
        
        # Negative samples processing
        # negative_words: (batch_size, num_negatives)
        neg_embeds = self.context_embeddings(negative_words)  # (batch_size, num_negatives, dim)
        center_embeds_expanded = center_embeds.unsqueeze(1)     # (batch_size, 1, dim)
        neg_score = torch.bmm(neg_embeds, center_embeds_expanded.transpose(1,2)).squeeze()  # (batch_size, num_negatives)
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)
        
        loss = - (pos_loss + neg_loss)
        return loss.mean()

def train_skipgram(embedding_dim=100, window_size=10, min_count=5, num_negatives=5, epochs=5, batch_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    sentences = load_brown_corpus()
    word2idx, idx2word, vocab = build_vocab(sentences, min_count)
    training_data = generate_training_data_skipgram(sentences, word2idx, window_size)
    vocab_size = len(word2idx)
    
    # Pre-compute negative sampling table
    table_size = 1000000
    neg_table = []
    neg_sampling_dist = get_negative_sampling_distribution(vocab, word2idx)
    for idx, prob in enumerate(neg_sampling_dist):
        neg_table.extend([idx] * int(prob * table_size))
    neg_table = np.array(neg_table)
    
    model = SkipGramModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.025)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # Convert training data to numpy arrays for faster processing
    center_words = np.array([pair[0] for pair in training_data])
    context_words = np.array([pair[1] for pair in training_data])
    
    num_batches = (len(training_data) + batch_size - 1) // batch_size
    indices = np.arange(len(training_data))
    
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0
        
        for i in range(num_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            center_batch = torch.LongTensor(center_words[batch_indices]).to(device)
            context_batch = torch.LongTensor(context_words[batch_indices]).to(device)
            
            # Fast negative sampling using pre-computed table
            neg_indices = np.random.choice(neg_table, size=(len(batch_indices), num_negatives))
            negative_batch = torch.LongTensor(neg_indices).to(device)
            
            optimizer.zero_grad()
            loss = model(center_batch, context_batch, negative_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 1000 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        scheduler.step()
    
    return model, word2idx, idx2word

if __name__ == "__main__":
    model, word2idx, idx2word = train_skipgram()
    # Save the center embeddings as the final word vectors
    torch.save({'embeddings': model.center_embeddings.weight.data, 'word2idx': word2idx, 'idx2word': idx2word}, "skipgram_2.pt")
    print("Skip-gram embeddings saved to skipgram_2.pt")
