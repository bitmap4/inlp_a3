# svd.py
import numpy as np
import torch
from helper import load_brown_corpus, build_vocab, get_cooccurrence_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def train_svd_embeddings(dim=100, window_size=10, min_count=5):
    sentences = load_brown_corpus()
    word2idx, idx2word, vocab = build_vocab(sentences, min_count)
    cooc_matrix = get_cooccurrence_matrix(sentences, word2idx, window_size)
    
    # Convert to sparse matrix and apply log scaling
    sparse_matrix = csr_matrix(cooc_matrix)
    sparse_matrix.data = np.log(sparse_matrix.data + 1)
    
    # Perform truncated SVD (much faster than full SVD)
    U, S, Vt = svds(sparse_matrix, k=dim)
    
    # Sort by singular values in descending order
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]
    
    # Compute embeddings
    embeddings = U * np.sqrt(S)[np.newaxis, :]
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings_tensor, word2idx, idx2word

if __name__ == "__main__":
    embeddings, word2idx, idx2word = train_svd_embeddings()
    torch.save({'embeddings': embeddings, 'word2idx': word2idx, 'idx2word': idx2word}, "svd.pt")
    print("SVD embeddings saved to svd.pt")
