import numpy as np
import torch
from nltk.corpus import brown, stopwords
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from config import *

# Load and tokenize the Brown corpus
corpus = brown.words()
tokens = [word.lower() for word in corpus if word.isalpha()]
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

# Build vocabulary
vocab = Counter(tokens)
vocab = {word: count for word, count in vocab.items()}
vocab_list = list(vocab.keys())
vocab_size = len(vocab_list)
word_to_idx = {word: i for i, word in enumerate(vocab_list)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Build Co-occurrence Matrix
co_matrix = np.zeros((vocab_size, vocab_size))
for i, word in enumerate(tokens):
    if word in word_to_idx:
        word_idx = word_to_idx[word]
        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)
        for j in range(start, end):
            if i != j and tokens[j] in word_to_idx:
                context_idx = word_to_idx[tokens[j]]
                co_matrix[word_idx, context_idx] += 1

# Apply SVD
svd = TruncatedSVD(n_components=EMBEDDING_DIM)
word_vectors = svd.fit_transform(co_matrix)

# Convert to PyTorch tensor
word_vectors_tensor = torch.tensor(word_vectors, dtype=torch.float32)

# Save embeddings
torch.save({'embeddings': word_vectors_tensor, 'word_to_idx': word_to_idx}, "./models/svd.pt")
print("SVD embeddings and word-to-index mapping saved successfully.")
