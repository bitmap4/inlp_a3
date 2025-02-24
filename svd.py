# svd.py
import numpy as np
import torch
from helper import load_brown_corpus, build_vocab, get_cooccurrence_matrix

def train_svd_embeddings(dim=300, window_size=5, min_count=5):
    sentences = load_brown_corpus()
    word2idx, idx2word, vocab = build_vocab(sentences, min_count)
    cooc_matrix = get_cooccurrence_matrix(sentences, word2idx, window_size)
    
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(cooc_matrix, full_matrices=False)
    
    # Compute embeddings as U * sqrt(S) (using only the top 'dim' components)
    S_sqrt = np.sqrt(S[:dim])
    embeddings = U[:, :dim] * S_sqrt[np.newaxis, :]
    
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)
    return embeddings_tensor, word2idx, idx2word

if __name__ == "__main__":
    embeddings, word2idx, idx2word = train_svd_embeddings()
    torch.save({'embeddings': embeddings, 'word2idx': word2idx, 'idx2word': idx2word}, "svd.pt")
    print("SVD embeddings saved to svd.pt")
