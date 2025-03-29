import pandas as pd
import ast  
from sklearn.naive_bayes import CategoricalNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
vectorizer = CountVectorizer(analyzer=lambda x: x)  
X_train_vectors = vectorizer.fit_transform(X_train).toarray()
X_test_vectors = vectorizer.transform(X_test).toarray()

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_train_discrete = discretizer.fit_transform(X_train_vectors)
X_test_discrete = discretizer.transform(X_test_vectors)
clf = CategoricalNB()
clf.fit(X_train_discrete, y_train)

y_pred = clf.predict(X_test_discrete)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: " + str(accuracy * 100) + "%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
