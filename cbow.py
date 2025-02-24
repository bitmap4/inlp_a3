import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import brown
from collections import Counter
from config import *

# Load and preprocess Brown corpus
corpus = brown.sents()
word_counts = Counter([word for sentence in corpus for word in sentence])
vocab = {word: i for i, (word, count) in enumerate(word_counts.items())}
idx_to_word = {i: word for word, i in vocab.items()}
vocab_size = len(vocab)

# Generate training data for CBOW
def generate_cbow_pairs(corpus, window_size):
    pairs = []
    for sentence in corpus:
        for i, word in enumerate(sentence):
            if word in vocab:
                context = []
                for j in range(-window_size, window_size + 1):
                    if j != 0 and 0 <= i + j < len(sentence) and sentence[i + j] in vocab:
                        context.append(vocab[sentence[i + j]])
                if len(context) == 2 * window_size:
                    pairs.append((context, vocab[word]))  # (context_words, target)
    return pairs

pairs = generate_cbow_pairs(corpus, WINDOW_SIZE)

# Convert pairs to tensors
context_tensors = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
target_tensors = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)

# CBOW Model with Negative Sampling
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.output_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)

    def forward(self, context_words, target, negative_samples):
        context_embeds = self.embeddings(context_words).mean(dim=1)
        target_embed = self.output_embeddings(target)
        neg_embed = self.output_embeddings(negative_samples)

        positive_score = torch.mul(context_embeds, target_embed).sum(dim=1).sigmoid()
        negative_score = torch.mul(context_embeds.unsqueeze(1), neg_embed).sum(dim=2).sigmoid()

        return -torch.log(positive_score + 1e-9).mean() - torch.log(1 - negative_score + 1e-9).mean()

# Generate negative samples
def get_negative_samples(batch_size, num_samples):
    neg_samples = torch.randint(0, vocab_size, (batch_size, num_samples))
    return neg_samples

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CBOWModel(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

context_tensors = context_tensors.to(device)
target_tensors = target_tensors.to(device)

prev_loss = float('inf')

for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(0, len(context_tensors), BATCH_SIZE):
        context_batch = context_tensors[i:i+BATCH_SIZE]
        target_batch = target_tensors[i:i+BATCH_SIZE]
        negative_samples = get_negative_samples(len(target_batch), NEGATIVE_SAMPLES).to(device)

        optimizer.zero_grad()
        loss = model(context_batch, target_batch, negative_samples)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    if prev_loss - total_loss < PATIENCE:
        print("Training converged.")
        break

    prev_loss = total_loss

# Save the trained embeddings and word-to-index mapping
embeddings = model.embeddings.weight.data.cpu()
torch.save({
    'embeddings': embeddings,
    'word_to_idx': vocab
}, "./models/cbow.pt")
print("CBOW embeddings and word-to-index mapping saved successfully.")

