import numpy as np 
import pandas as pd 
import os 

from sklearn.ensemble import GradientBoostingClassifier  

 

train_data = pd.read_csv('./data/features/train_bow.csv')
x_train = train_data.iloc[:,:-1].values  
y_train = train_data.iloc[:,-1].values  

clf = GradientBoostingClassifier(n_estimators=50)
clf.fit(x_train,y_train)

# save the model 
import pickle 
pickle.dump(clf,open('model.pkl','wb'))
