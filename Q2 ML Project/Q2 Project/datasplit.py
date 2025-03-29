import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

df = pd.read_csv('spam2_cleaned_tokenized.csv')

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_data.to_csv('Q2train_data2.csv', index=False)
test_data.to_csv('Q2test_data2.csv', index=False)

