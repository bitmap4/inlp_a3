# cbow.py
import torch
import torch.nn as nn
import torch.optim as optim
from helper import load_brown_corpus, build_vocab, generate_training_data_cbow, get_negative_sampling_distribution, sample_negative_examples
import random

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

def train_cbow(embedding_dim=100, window_size=10, min_count=5, num_negatives=5, epochs=5, batch_size=128):
    sentences = load_brown_corpus()
    word2idx, idx2word, vocab = build_vocab(sentences, min_count)
    training_data = generate_training_data_cbow(sentences, word2idx, window_size)
    vocab_size = len(word2idx)
    
    model = CBOWModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    neg_sampling_dist = get_negative_sampling_distribution(vocab, word2idx)
    
    for epoch in range(epochs):
        random.shuffle(training_data)
        losses = []
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i: i+batch_size]
            context_batch = torch.tensor([sample[0] for sample in batch], dtype=torch.long)
            target_batch = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
            negative_batch = []
            for sample in batch:
                negatives = sample_negative_examples(neg_sampling_dist, num_negatives, sample[1])
                negative_batch.append(negatives)
            negative_batch = torch.tensor(negative_batch, dtype=torch.long)
            
            optimizer.zero_grad()
            loss = model(context_batch, target_batch, negative_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {sum(losses)/len(losses)}")
    
    return model, word2idx, idx2word

if __name__ == "__main__":
    model, word2idx, idx2word = train_cbow()
    # Save the input embeddings as the final word vectors
    torch.save({'embeddings': model.input_embeddings.weight.data, 'word2idx': word2idx, 'idx2word': idx2word}, "cbow.pt")
    print("CBOW embeddings saved to cbow.pt")
