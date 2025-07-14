import numpy as np 
import pandas as pd 
import pickle 

from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
 
clf = pickle.load(open('model.pkl','rb'))  
test_df = pd.read_csv('./data/features/test_bow.csv')
x_test = test_df.iloc[:,:-1].values
y_test = test_df.iloc[:,-1].values 


y_pred = clf.predict(x_test)
y_pred_prob = clf.predict_proba(x_test)[:,1]

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
auc = roc_auc_score(y_test,y_pred_prob)

metric_dic = {
    'aaccuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc 
}

import json 
with open('metrics.json','w') as file:
    json.dump(metric_dic,file,indent=4)

