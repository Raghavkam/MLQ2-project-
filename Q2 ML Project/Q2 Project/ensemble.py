import pandas as pd
import numpy as np
import ast
from collections import Counter
from math import log
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


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


embedding_path = "GoogleNews-vectors-negative300.bin"
try:
    word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    print("Pretrained Word2Vec embeddings loaded successfully.")
except FileNotFoundError:
    print("Embedding file not found. Using simple word frequency instead.")
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

#
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


def predict_word2vec(words):
    spam_score = prior_spam
    ham_score = prior_ham

    for word in words:
        spam_score += log(compute_word_prob(word, "spam"))
        ham_score += log(compute_word_prob(word, "ham"))

    return 1 if spam_score > ham_score else 0


word_counts_tfidf = {"spam": Counter(), "ham": Counter()}
doc_counts_tfidf = {"spam": 0, "ham": 0}
doc_freq = Counter()
doc_total = len(X_train)


for i, words in enumerate(X_train):
    label = "spam" if y_train[i] == 1 else "ham"
    doc_counts_tfidf[label] += 1
    word_counts_tfidf[label].update(words)
    vocab.update(words)
    for word in set(words):  
        doc_freq[word] += 1


def compute_tfidf(word, words, label):
    if word in word_counts_tfidf[label]:
        tf = words.count(word) / len(words)  
        idf = log((1 + doc_total) / (1 + doc_freq[word])) + 1  
        return tf * idf
    return 0


def predict_tfidf(words):
    spam_score = prior_spam
    ham_score = prior_ham

    for word in words:
        if word in vocab:
            spam_score += log(compute_tfidf(word, words, "spam") + 1e-9)
            ham_score += log(compute_tfidf(word, words, "ham") + 1e-9)

    return 1 if spam_score > ham_score else 0


vectorizer_bayes = CountVectorizer(analyzer=lambda x: x, max_features=5)
X_train_vectors = vectorizer_bayes.fit_transform(X_train).toarray()
X_test_vectors = vectorizer_bayes.transform(X_test).toarray()

X_train_df = pd.DataFrame(X_train_vectors, columns=[f'word_{i}' for i in range(X_train_vectors.shape[1])])
X_train_df['Spam'] = y_train
X_test_df = pd.DataFrame(X_test_vectors, columns=[f'word_{i}' for i in range(X_test_vectors.shape[1])])
X_test_df['Spam'] = y_test

dependency_graph = BayesianNetwork([
    ('word_0', 'Spam'), ('word_1', 'Spam'), ('word_2', 'Spam'),
    ('word_3', 'word_0'), ('word_4', 'word_1')
])
dependency_graph.fit(X_train_df, estimator=MaximumLikelihoodEstimator)
inference = VariableElimination(dependency_graph)

def classify_instance(instance):
    evidence = {f'word_{i}': int(instance[i]) for i in range(len(instance))}
    try:
        result = inference.map_query(variables=['Spam'], evidence=evidence, show_progress=False)
        return result['Spam']
    except IndexError:
        return np.bincount(y_train).argmax()


vectorizer = CountVectorizer(analyzer=lambda x: x)
X_train_vectors = vectorizer.fit_transform(X_train).toarray()
X_test_vectors = vectorizer.transform(X_test).toarray()

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_train_discrete = discretizer.fit_transform(X_train_vectors)
X_test_discrete = discretizer.transform(X_test_vectors)

clf = CategoricalNB()
clf.fit(X_train_discrete, y_train)
y_pred_nb = clf.predict(X_test_discrete)  


y_pred_word2vec = [predict_word2vec(words) for words in X_test]
y_pred_tfidf = [predict_tfidf(words) for words in X_test]
y_pred_dependency = X_test_df.drop(columns=['Spam']).apply(classify_instance, axis=1)

'''
sum = 0
for i in y_pred_nb:
    if i == 1:
        sum += 1

print(sum)

sum = 0
'''


def weighted_voting(w2v, tfidf, dependency, nb):
    if nb == 1:
        print(1)
        return 1  

   # if nb != 1
    # Otherwise, use weighted voting (favoring TF-IDF)
    weights = {"w2v": .3, "tfidf": .5, "bayesian": .05, "nb": .15}  
    score = (w2v * weights["w2v"]) + (tfidf * weights["tfidf"]) + (dependency * weights["bayesian"]) + (nb * weights["nb"])

    return 1 if score > 0.45 else 0  

y_pred_weighted = [
     weighted_voting(y2, y1, y3, y4)  
    for y1, y2, y3, y4 in zip(y_pred_word2vec, y_pred_tfidf, y_pred_dependency, y_pred_nb)
]



def print_model_performance(name, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100

    TP, FP, FN, TN = confusion_matrix(y_test, y_pred).ravel()
    print(f"\n{name} Model Performance:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(confusion_matrix(y_test, y_pred))


print_model_performance("Word2Vec", y_pred_word2vec)
print_model_performance("TF-IDF", y_pred_tfidf)
print_model_performance("Bayesian Network", y_pred_dependency)
print_model_performance("SKLearn", y_pred_nb)
print_model_performance("Weighted Voting Ensemble", y_pred_weighted)
