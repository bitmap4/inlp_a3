# wordsim.py
import torch
import csv
import argparse
import os
import numpy as np
import torch.nn.functional as F
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def load_embeddings(model_file):
    data = torch.load(model_file)
    embeddings = data['embeddings']
    word2idx = data['word2idx']
    idx2word = data['idx2word']
    return embeddings, word2idx, idx2word

def load_wordsim(file_path):
    pairs = []
    human_scores = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the first row (header)

        # Ensure the header has the correct number of columns
        header = [col.strip() for col in header]  # Strip spaces from header
        if len(header) < 3:
            raise ValueError("CSV file format is incorrect: Expected at least 3 columns (Word1, Word2, Score).")

        for row in reader:
            if len(row) < 3:
                continue  # Skip invalid rows
            word1, word2, score = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                score = float(score)
                pairs.append((word1, word2))
                human_scores.append(score)
            except ValueError:
                continue  # Skip rows with non-numeric scores

    return pairs, human_scores


def compute_cosine_similarity(embeddings, word2idx, word1, word2):
    if word1 not in word2idx or word2 not in word2idx:
        return None
    vec1 = embeddings[word2idx[word1]]
    vec2 = embeddings[word2idx[word2]]
    cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    return cos_sim.item()

def evaluate_wordsim(model_file, wordsim_file, plot=False):
    embeddings, word2idx, idx2word = load_embeddings(model_file)
    pairs, human_scores = load_wordsim(wordsim_file)
    computed_scores = []
    filtered_human = []
    filtered_pairs = []
    
    # Create output CSV filename
    output_file = f"results/predicted_{os.path.basename(model_file)}.csv"
    
    # Store results for CSV
    results = []
    for (w1, w2), human_score in zip(pairs, human_scores):
        sim = compute_cosine_similarity(embeddings, word2idx, w1.lower(), w2.lower())
        if sim is not None:
            computed_scores.append(sim)
            filtered_human.append(human_score)
            filtered_pairs.append((w1, w2))
            results.append([w1, w2, sim])
    
    # Save predictions to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Word1', 'Word2', 'Predicted'])
        writer.writerows(results)
    print(f"Predictions saved to {output_file}")
    
    corr, _ = spearmanr(filtered_human, computed_scores)
    print(f"Spearman Rank Correlation for {model_file}: {corr}")
    
    if plot:
        plt.figure()
        plt.scatter(filtered_human, computed_scores, alpha=0.5)
        plt.xlabel("Human Similarity Scores")
        plt.ylabel("Cosine Similarity")
        plt.title(f"WordSim Evaluation - {os.path.basename(model_file)}")
        os.makedirs("results", exist_ok=True)
        plot_file = os.path.join("results", f"wordsim_{os.path.basename(model_file)}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="svd.pt", help="Path to the embedding model file")
    parser.add_argument("--wordsim", type=str, default="wordsim353crowd.csv", help="Path to the WordSim-353 dataset file")
    parser.add_argument("--plot", action="store_true", help="Plot the results and save to the results folder")
    args = parser.parse_args()
    evaluate_wordsim(args.model, args.wordsim, args.plot)
    # Sample usage : python wordsim.py --model skipgram.pt --wordsim "WordSim353 Crowd.csv" --plot
