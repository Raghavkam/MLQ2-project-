import pandas as pd
import numpy as np
import ast  
from collections import Counter
from math import log

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

word_counts = {"spam": Counter(), "ham": Counter()}
doc_counts = {"spam": 0, "ham": 0}
doc_freq = Counter()
doc_total = len(X_train)
vocab = set()
for i, words in enumerate(X_train):
    label = "spam" if y_train[i] == 1 else "ham"
    doc_counts[label] += 1
    word_counts[label].update(words)
    vocab.update(words)
    for word in set(words): 
        doc_freq[word] += 1

def compute_tfidf(word, words, label):
    if word in word_counts[label]: 
        tf = words.count(word) / len(words)  
        idf = log((1 + doc_total) / (1 + doc_freq[word])) + 1  
        return tf * idf
    return 0 

prior_spam = log(doc_counts["spam"] / doc_total) if doc_counts["spam"] > 0 else -np.inf
prior_ham = log(doc_counts["ham"] / doc_total) if doc_counts["ham"] > 0 else -np.inf

alpha = 1  
V = len(vocab) 
def predict(words):
    spam_score = prior_spam
    ham_score = prior_ham

    for word in words:
        if word in vocab:
            spam_score += log(compute_tfidf(word, words, "spam") + 1e-9) 
            ham_score += log(compute_tfidf(word, words, "ham") + 1e-9)

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