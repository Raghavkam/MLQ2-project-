import pandas as pd
import numpy as np
import ast  
from collections import Counter
from math import log
from gensim.models import KeyedVectors 

df_train = pd.read_csv("Q2train_data2.csv", encoding="utf-8")
df_test = pd.read_csv("Q2test_data2.csv", encoding="utf-8")
if isinstance(df_train["v2"].iloc[0], str):
    df_train["v2"] = df_train["v2"].apply(ast.literal_eval)
if isinstance(df_test["v2"].iloc[0], str):
    df_test["v2"] = df_test["v2"].apply(ast.literal_eval)

X_train = df_train["v2"].tolist()  
y_train = df_train["v1"].tolist()  

X_test = df_test["v2"].tolist()  
y_test = df_test["v1"].tolist()  

embedding_path = "/mnt/data/GoogleNews-vectors-negative300.bin"  
try:
    word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    print("Pretrained embeddings loaded successfully.")
except FileNotFoundError:
    print("Embedding file not found. Ensure you have a Word2Vec or GloVe model available.")
    word_vectors = None 
word_counts = {"spam": Counter(), "ham": Counter()}
total_words = {"spam": 0, "ham": 0}
doc_counts = {"spam": 0, "ham": 0}
vocab = set()

for i, words in enumerate(X_train):
    label = "spam" if y_train[i] == 1 else "ham"
    doc_counts[label] += 1
    word_counts[label].update(words)
    total_words[label] += len(words)
    vocab.update(words)

prior_spam = log(doc_counts["spam"] / len(y_train))
prior_ham = log(doc_counts["ham"] / len(y_train))

alpha = 1  
V = len(vocab)  
def compute_word_prob(word, label):
    if word_vectors and word in word_vectors:
        class_words = word_counts[label].keys()
        similarity_scores = [
            word_vectors.similarity(word, class_word) for class_word in class_words if class_word in word_vectors
        ]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        return (avg_similarity + alpha) / (total_words[label] + alpha * V + 1e-9)
    else:
        return (word_counts[label][word] + alpha) / (total_words[label] + alpha * V + 1e-9)

def predict(words):
    spam_score = prior_spam
    ham_score = prior_ham

    for word in words:
        spam_score += log(compute_word_prob(word, "spam"))
        ham_score += log(compute_word_prob(word, "ham"))

    return 1 if spam_score > ham_score else 0  
y_pred = [predict(words) for words in X_test]

def evaluate(y_true, y_pred):
    TP = FP = TN = FN = 0
    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == 1 and predicted_label == 1:
            TP += 1
        elif true_label == 0 and predicted_label == 1:
            FP += 1
        elif true_label == 1 and predicted_label == 0:
            FN += 1
        elif true_label == 0 and predicted_label == 0:
            TN += 1

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    return accuracy, TP, FN, FP, TN

accuracy, TP, FN, FP, TN = evaluate(y_test, y_pred)

print("Accuracy: " + str(accuracy * 100) + "%")
print("Confusion Matrix:")
print("TP: " + str(TP) + ", FP: " + str(FP))
print("FN: " + str(FN) + ", TN: " + str(TN))