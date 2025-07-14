import numpy as np 
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

 

train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

train_data.fillna('',inplace=True)
test_data.fillna('',inplace=True)

# apply bow 
X_train = train_data['content'].values 
y_train = train_data['sentiment'].values 

X_test = test_data['content'].values 
y_test = test_data['sentiment'].values  

vectorizer = CountVectorizer(max_features = 70)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train 

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test 

#store data inside data/features
data_path = os.path.join("data","features")
os.makedirs(data_path)

train_df.to_csv(os.path.join(data_path,"train_bow.csv"))
test_df.to_csv(os.path.join(data_path,"test_bow.csv"))
