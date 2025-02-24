import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import brown, stopwords
import random
from collections import Counter
from config import *
from tqdm import tqdm

# Load and preprocess Brown corpus
corpus = brown.sents()
stop_words = stop_words = set(stopwords.words('english'))
# Build vocabulary
word_counts = Counter([word for sentence in corpus for word in sentence if word.lower() not in stop_words])
vocab = {word: i for i, (word, count) in enumerate(word_counts.items())}
idx_to_word = {i: word for word, i in vocab.items()}
vocab_size = len(vocab)

# Generate training data for Skip-Gram model
def generate_skipgram_pairs(corpus, window_size):
    pairs = []
    for sentence in corpus:
        for i, word in enumerate(sentence):
            if word in vocab:
                center_word = vocab[word]
                context_words = []
                for j in range(-window_size, window_size + 1):
                    if j != 0 and 0 <= i + j < len(sentence) and sentence[i + j] in vocab:
                        context_words.append(vocab[sentence[i + j]])
                for context in context_words:
                    pairs.append((center_word, context))
    return pairs

pairs = generate_skipgram_pairs(corpus, WINDOW_SIZE)
pairs = torch.tensor(pairs, dtype=torch.long)

# Skip-Gram with Negative Sampling Model
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.center_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)

    def forward(self, center, context, negative_samples):
        center_embed = self.center_embeddings(center)
        context_embed = self.context_embeddings(context)
        neg_embed = self.context_embeddings(negative_samples)

        positive_score = torch.mul(center_embed, context_embed).sum(dim=1).sigmoid()
        negative_score = torch.mul(center_embed.unsqueeze(1), neg_embed).sum(dim=2).sigmoid()

        return -torch.log(positive_score + 1e-9).mean() - torch.log(1 - negative_score + 1e-9).mean()

# Generate negative samples
def get_negative_samples(batch_size, num_samples):
    neg_samples = torch.randint(0, vocab_size, (batch_size, num_samples))
    return neg_samples

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

pairs = pairs.to(device)

previous_loss = float('inf')

for epoch in range(EPOCHS):
    random.shuffle(pairs)
    total_loss = 0
    # Create progress bar for batches
    progress_bar = tqdm(range(0, len(pairs), BATCH_SIZE), desc=f'Epoch {epoch+1}/{EPOCHS}')
    
    for i in progress_bar:
        batch = pairs[i:i+BATCH_SIZE]
        center_words, context_words = batch[:, 0], batch[:, 1]
        negative_samples = get_negative_samples(len(batch), NEGATIVE_SAMPLES).to(device)

        optimizer.zero_grad()
        loss = model(center_words, context_words, negative_samples)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    print(f"Epoch {epoch+1}/{EPOCHS}, Total Loss: {total_loss:.4f}")

    if previous_loss - total_loss < PATIENCE:
        print("Training converged.")
        break

    previous_loss = total_loss

# Extract embeddings and convert to list
embeddings = model.center_embeddings.weight.data.cpu()
torch.save({
    'embeddings': embeddings,
    'word_to_idx': vocab
}, "./models/skipgram.pt")
print("Skipgram Embeddings and word-to-index mapping saved successfully.")
