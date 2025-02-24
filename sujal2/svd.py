# svd.py
import numpy as np
import torch
from helper import load_brown_corpus, build_vocab, get_cooccurrence_matrix
from tqdm import tqdm
import logging
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import normalize

def train_svd_embeddings(
    dim=300, 
    window_size=5, 
    min_count=5,
    normalize_embeddings=True,
    batch_size=1000
):
    """
    Train SVD embeddings with improved memory efficiency and performance.
    
    Args:
        dim: Embedding dimension
        window_size: Context window size
        min_count: Minimum word frequency
        normalize_embeddings: Whether to L2-normalize the embeddings
        batch_size: Batch size for processing
    """
    logging.info("Loading and preprocessing corpus...")
    try:
        sentences = load_brown_corpus()
        word2idx, idx2word, vocab = build_vocab(sentences, min_count)
        
        logging.info("Building co-occurrence matrix...")
        cooc_matrix = get_cooccurrence_matrix(sentences, word2idx, window_size)
        
        # Convert to sparse matrix if dense
        if not issparse(cooc_matrix):
            cooc_matrix = csr_matrix(cooc_matrix)
        
        logging.info("Performing SVD...")
        # Use sparse SVD for better memory efficiency
        U, S, Vt = svds(cooc_matrix, k=dim)
        
        # Process embeddings in batches to save memory
        embeddings = []
        S_sqrt = np.sqrt(S)
        
        for i in tqdm(range(0, U.shape[0], batch_size), desc="Computing embeddings"):
            batch = U[i:i+batch_size, :] * S_sqrt[np.newaxis, :]
            if normalize_embeddings:
                batch = normalize(batch)
            embeddings.append(batch)
            
        embeddings = np.vstack(embeddings)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)
        
        return embeddings_tensor, word2idx, idx2word
        
    except Exception as e:
        logging.error(f"Error during SVD training: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        embeddings, word2idx, idx2word = train_svd_embeddings()
        
        output_file = "svd.pt"
        torch.save(
            {
                'embeddings': embeddings,
                'word2idx': word2idx,
                'idx2word': idx2word
            },
            output_file
        )
        logging.info(f"SVD embeddings saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Failed to train or save embeddings: {str(e)}")