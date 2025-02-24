# helper.py
import nltk
from nltk.corpus import brown
import numpy as np
import torch
from collections import Counter
import random

def download_corpus():
    nltk.download('brown')

def load_brown_corpus():
    # Ensure the corpus is downloaded
    try:
        sentences = brown.sents()
    except LookupError:
        download_corpus()
        sentences = brown.sents()
    # Convert words to lower case
    sentences = [[word.lower() for word in sent] for sent in sentences]
    return sentences

def build_vocab(sentences, min_count=5):
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(sent)
    # Filter out infrequent words
    vocab = {word: count for word, count in word_counts.items() if count >= min_count}
    idx2word = list(vocab.keys())
    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    return word2idx, idx2word, vocab

def get_cooccurrence_matrix(sentences, word2idx, window_size=2):
    vocab_size = len(word2idx)
    cooc_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for sent in sentences:
        indices = [word2idx[word] for word in sent if word in word2idx]
        for center_i, center_word in enumerate(indices):
            start = max(0, center_i - window_size)
            end = min(len(indices), center_i + window_size + 1)
            for context_i in range(start, end):
                if context_i != center_i:
                    context_word = indices[context_i]
                    cooc_matrix[center_word, context_word] += 1.0
    return cooc_matrix

def generate_training_data_skipgram(sentences, word2idx, window_size=2):
    data = []
    for sent in sentences:
        indices = [word2idx[word] for word in sent if word in word2idx]
        for i, center in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context = indices[j]
                    data.append((center, context))
    return data

def generate_training_data_cbow(sentences, word2idx, window_size=2):
    data = []
    for sent in sentences:
        indices = [word2idx[word] for word in sent if word in word2idx]
        # Use a sliding window where the target is in the center
        for i in range(window_size, len(indices) - window_size):
            context = indices[i - window_size:i] + indices[i+1:i+window_size+1]
            target = indices[i]
            data.append((context, target))
    return data

def get_negative_sampling_distribution(vocab, word2idx, power=0.75):
    word_freqs = np.zeros(len(word2idx))
    for word, idx in word2idx.items():
        word_freqs[idx] = vocab[word]
    # Smooth the distribution
    word_freqs = word_freqs ** power
    word_freqs = word_freqs / np.sum(word_freqs)
    return word_freqs

def sample_negative_examples(word_freqs, num_samples, exclude):
    negatives = []
    while len(negatives) < num_samples:
        neg = np.random.choice(len(word_freqs), p=word_freqs)
        if neg != exclude:
            negatives.append(neg)
    return negatives
