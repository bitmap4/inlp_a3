import csv
import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from config import *

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def load_dataset(filepath):
    word_pairs = []
    human_scores = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            word1, word2, score = row
            word_pairs.append((word1, word2))
            human_scores.append(float(score))
            normalized_human_scores = []
            for i in human_scores:
                normalized_human_scores.append(i/SIMILARITY_SCORE_NORMALIZER_FACTOR)
    return word_pairs, normalized_human_scores

def compute_cosine_similarities(word_pairs, vectors, indices):
    similarities = []
    for word1, word2 in word_pairs:
        try:
            vec1 = vectors[indices[word1]]
            vec2 = vectors[indices[word2]]
            similarity = cosine_similarity(vec1, vec2)
            similarities.append(similarity)
        except:
            similarities.append(None)
    return similarities

def filter_valid_pairs(human_scores, similarities, embedding):
    valid_human_scores = []
    valid_similarities = []
    for human_score, similarity in zip(human_scores, similarities):
        if similarity is not None:
            valid_human_scores.append(human_score)
            valid_similarities.append(similarity)
    plt.scatter(valid_human_scores, valid_similarities)
    plt.xlabel('Human Scores')
    plt.ylabel('Cosine Similarities')
    plt.title('Human Scores vs Cosine Similarities')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.axvline(x=10, color='b', linestyle='--')
    plt.axhline(y=0, color='k', linestyle='--')
    if embedding == "SVD":
        plt.savefig('./plots/svd.png')
    elif embedding == "CBOW":
        plt.savefig('./plots/cbow.png')
    elif embedding == "Skipgram":
        plt.savefig('./plots/skipgram.png')
    return valid_human_scores, valid_similarities

def main():
    embedding = input("Enter the embedding type (SVD or CBOW or Skipgram): ")
    if embedding == "SVD":
        word_vectors = torch.load('./models/svd.pt')
    elif embedding == "CBOW":
        word_vectors = torch.load('./models/cbow.pt')
    elif embedding == "Skipgram":
        word_vectors = torch.load('./models/skipgram.pt')
    vectors  = word_vectors['embeddings']
    indices = word_vectors["word_to_idx"]
    filepath = 'wordsim353crowd.csv'
    word_pairs, human_scores = load_dataset(filepath)
    similarities = compute_cosine_similarities(word_pairs, vectors, indices)
    valid_human_scores, valid_similarities = filter_valid_pairs(human_scores, similarities, embedding)
    spearman_corr, _ = spearmanr(valid_human_scores, valid_similarities)
    print(f"Spearman's Rank Correlation for {embedding}: {spearman_corr}")

if __name__ == "__main__":
    main()
