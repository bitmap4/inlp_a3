# skipgram.py
import torch
import torch.nn as nn
import torch.optim as optim
from helper import load_brown_corpus, build_vocab, generate_training_data_skipgram, get_negative_sampling_distribution, sample_negative_examples
import random
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

def train_skipgram(embedding_dim=100, window_size=10, min_count=5, num_negatives=5, epochs=5, batch_size=128):
    sentences = load_brown_corpus()
    word2idx, idx2word, vocab = build_vocab(sentences, min_count)
    training_data = generate_training_data_skipgram(sentences, word2idx, window_size)
    vocab_size = len(word2idx)
    
    model = SkipGramModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    neg_sampling_dist = get_negative_sampling_distribution(vocab, word2idx)
    # loop = tqdm(range(epochs), total=epochs)
    for epoch in range(epochs):
        random.shuffle(training_data)
        losses = []
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i: i+batch_size]
            center_batch = torch.tensor([pair[0] for pair in batch], dtype=torch.long)
            context_batch = torch.tensor([pair[1] for pair in batch], dtype=torch.long)
            # For each positive pair, sample negatives (excluding the context word)
            negative_batch = []
            for pair in batch:
                negatives = sample_negative_examples(neg_sampling_dist, num_negatives, pair[1])
                negative_batch.append(negatives)
            negative_batch = torch.tensor(negative_batch, dtype=torch.long)
            
            optimizer.zero_grad()
            loss = model(center_batch, context_batch, negative_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {sum(losses)/len(losses)}")
    
    return model, word2idx, idx2word

if __name__ == "__main__":
    model, word2idx, idx2word = train_skipgram()
    # Save the center embeddings as the final word vectors
    torch.save({'embeddings': model.center_embeddings.weight.data, 'word2idx': word2idx, 'idx2word': idx2word}, "skipgram_2.pt")
    print("Skip-gram embeddings saved to skipgram_2.pt")
