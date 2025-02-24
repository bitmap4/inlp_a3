# Word Embedding Training and Evaluation

## 1. Final Submission Contents

The final submission contains the following files:

### Training Scripts
- **svd.py**: Trains word embeddings using the Singular Value Decomposition (SVD) method and saves the word vectors.
- **skipgram.py**: Trains word embeddings using the Skip-gram method (with negative sampling) and saves the word vectors.
- **cbow.py**: Trains word embeddings using the Continuous Bag of Words (CBOW) method (with negative sampling) and saves the word vectors.

### Saved Word Vectors
- **svd.pt**: Saved word vectors for the entire vocabulary trained using SVD.
- **skipgram.pt**: Saved word vectors for the entire vocabulary trained using Skip-gram (with negative sampling).
- **cbow.pt**: Saved word vectors for the entire vocabulary trained using CBOW (with negative sampling).

### Additional Files
- **results/**: A folder containing plots as referenced in the report.
- **wordsim.py**: Contains the code for evaluating word similarity using the WordSim-353 dataset.
- **helper.py**: Contains helper functions for loading and processing the dataset.

---

## 2. Training the Embeddings

To train the word embeddings using different methods, run the following scripts:
```sh
python3 svd.py
python3 skipgram.py
python3 cbow.py
```
Each script generates a corresponding `.pt` file containing the trained word vectors.

---

## 3. Evaluating on WordSim-353

### Preparation
1. Place the **WordSim-353 dataset** file (e.g., `WordSim353 Crowd.csv`) in the project folder.
2. This file can be found in the assignment question document.

### Running Evaluation
To evaluate a trained word embedding model, use the following command:
```sh
python3 wordsim.py --model <path_to_model.pt> --wordsim "WordSim353 Crowd.csv" --plot 
```
Replace `<path_to_model.pt>` with the specific trained model file (e.g., `svd.pt`, `skipgram.pt`, or `cbow.pt`). This script:
- Computes the **Spearman correlation** between the trained embeddings and the WordSim-353 dataset.
- Produces a **scatter plot** saved in the `results/` folder.

---

## Hyperparameters Used
- **Window size**: 5 (for all three methods, here the Window size determines the context words surrounding a given center word.)
- **Embedding dimensions**: 300 (as it was seen that lower dimensions resulted in poor Spearman correlation)

---

## Dependencies
Ensure you have the required dependencies installed before running the scripts:
```sh
pip install numpy torch matplotlib pandas
```

---

## Author
Sujal Deoda



---

This README provides all the necessary instructions to train and evaluate word embeddings using multiple methods.


for cbow:
for d = 300, as num of negatives changed from 5 to 10 spearman correlation dropped from 0.145 to 0.135, ws = 5 
ws=3 ke liyewo badh gya 0.56 now

for d = 512, 0.137
for d = 100, 0.104


