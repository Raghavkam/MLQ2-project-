from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import ast  
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import logging

logging.getLogger("pgmpy").setLevel(logging.CRITICAL)
df_train = pd.read_csv('Q2train_data2.csv')
df_test = pd.read_csv('Q2test_data2.csv')

if isinstance(df_train["v2"].iloc[0], str):
    df_train["v2"] = df_train["v2"].apply(ast.literal_eval)
if isinstance(df_test["v2"].iloc[0], str):
    df_test["v2"] = df_test["v2"].apply(ast.literal_eval)

X_train = df_train["v2"].tolist()  
y_train = df_train["v1"].tolist()  

X_test = df_test["v2"].tolist()  
y_test = df_test["v1"].tolist()  

vectorizer = CountVectorizer(analyzer=lambda x: x, max_features=5)  
X_train_vectors = vectorizer.fit_transform(X_train).toarray()
X_test_vectors = vectorizer.transform(X_test).toarray()

X_train_df = pd.DataFrame(X_train_vectors, columns=['word_' + str(i) for i in range(X_train_vectors.shape[1])])
X_train_df['Spam'] = y_train  

X_test_df = pd.DataFrame(X_test_vectors, columns=['word_' + str(i) for i in range(X_test_vectors.shape[1])])
X_test_df['Spam'] = y_test

dependency_graph = BayesianNetwork([
    ('word_0', 'Spam'), 
    ('word_1', 'Spam'), 
    ('word_2', 'Spam'), 
    ('word_3', 'word_0'), 
    ('word_4', 'word_1')
])

dependency_graph.fit(X_train_df, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(dependency_graph)

def classify_instance(instance):
    evidence = {'word_' + str(i): int(instance[i]) for i in range(len(instance))}
    
    try:
        result = inference.map_query(variables=['Spam'], evidence=evidence, show_progress=False)
        return result['Spam']
    except IndexError:
        return np.bincount(y_train).argmax()  

predictions = X_test_df.drop(columns=['Spam']).apply(classify_instance, axis=1)

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

accuracy, TP, FN, FP, TN = evaluate(y_test, predictions)

print("Accuracy: " + str(accuracy * 100) + "%")
print("Confusion Matrix:")
print("TP: " + str(TP) + ", FP: " + str(FP))
print("FN: " + str(FN) + ", TN: " + str(TN))