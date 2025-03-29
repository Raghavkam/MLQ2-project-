import pandas as pd
import re

file_path = "spam2.csv"  
df = pd.read_csv(file_path, encoding='latin-1')
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})

def clean_and_tokenize(text):
    text = text.lower()  
    text = re.sub(r'\W+', ' ', text)  
    text = re.sub(r'\d+', '', text)  
    tokens = text.strip().split()
    return tokens
df['v2'] = df['v2'].apply(clean_and_tokenize)
df.to_csv("spam2_cleaned_tokenized.csv", index=False)

print(df.head())
