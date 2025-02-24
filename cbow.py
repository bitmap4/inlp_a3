# cbow.py
import torch
import torch.nn as nn
import torch.optim as optim
from helper import load_brown_corpus, build_vocab, generate_training_data_cbow, get_negative_sampling_distribution, sample_negative_examples
import random
import numpy as np

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, context_words, target_words, negative_words):
        # context_words: (batch_size, context_size)
        context_embeds = self.input_embeddings(context_words)  # (batch_size, context_size, dim)
        context_embeds = torch.mean(context_embeds, dim=1)       # (batch_size, dim)
        target_embeds = self.output_embeddings(target_words)    # (batch_size, dim)
        
        pos_score = torch.sum(context_embeds * target_embeds, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)
        
        # Negative samples: negative_words shape (batch_size, num_negatives)
        neg_embeds = self.output_embeddings(negative_words)     # (batch_size, num_negatives, dim)
        context_embeds_expanded = context_embeds.unsqueeze(1)     # (batch_size, 1, dim)
        neg_score = torch.bmm(neg_embeds, context_embeds_expanded.transpose(1,2)).squeeze()  # (batch_size, num_negatives)
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)
        
        loss = - (pos_loss + neg_loss)
        return loss.mean()

def train_cbow(embedding_dim=100, window_size=10, min_count=5, num_negatives=5, epochs=5, batch_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    sentences = load_brown_corpus()
    word2idx, idx2word, vocab = build_vocab(sentences, min_count)
    training_data = generate_training_data_cbow(sentences, word2idx, window_size)
    vocab_size = len(word2idx)
    
    # Pre-compute negative sampling table
    table_size = 1000000
    neg_table = []
    neg_sampling_dist = get_negative_sampling_distribution(vocab, word2idx)
    for idx, prob in enumerate(neg_sampling_dist):
        neg_table.extend([idx] * int(prob * table_size))
    neg_table = np.array(neg_table)
    
    model = CBOWModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.025)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # Convert training data to numpy arrays for faster processing
    context_words = np.array([sample[0] for sample in training_data])
    target_words = np.array([sample[1] for sample in training_data])
    
    num_batches = (len(training_data) + batch_size - 1) // batch_size
    indices = np.arange(len(training_data))
    
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0
        
        for i in range(num_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            context_batch = torch.LongTensor(context_words[batch_indices]).to(device)
            target_batch = torch.LongTensor(target_words[batch_indices]).to(device)
            
            # Fast negative sampling using pre-computed table
            neg_indices = np.random.choice(neg_table, size=(len(batch_indices), num_negatives))
            negative_batch = torch.LongTensor(neg_indices).to(device)
            
            optimizer.zero_grad()
            loss = model(context_batch, target_batch, negative_batch)
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
    model, word2idx, idx2word = train_cbow()
    # Save the input embeddings as the final word vectors
    torch.save({
        'embeddings': model.input_embeddings.weight.data.cpu(),
        'word2idx': word2idx,
        'idx2word': idx2word
    }, "cbow.pt")
    print("CBOW embeddings saved to cbow.pt")

if __name__ == "__main__":
    model, word2idx, idx2word = train_cbow()
    # Save the input embeddings as the final word vectors
    torch.save({'embeddings': model.input_embeddings.weight.data, 'word2idx': word2idx, 'idx2word': idx2word}, "cbow.pt")
    print("CBOW embeddings saved to cbow.pt")
