import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score,accuracy_score,classification_report
import joblib

data = pd.read_csv(r'E:\1.DEEPAK Data Science\GITREPO\DMT_MASTER_REPO\HeartAttack_LR\bmd.csv')
print (data.info())
Y = data.fracture
print (Y)
sex_medication = pd.get_dummies(data = data, columns = ['sex','medication'])
data = pd.concat([data,sex_medication],axis = 1)
data= data.drop(['sex','medication','fracture','id'], axis = 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

X_train, X_test, Y_train, Y_test = train_test_split(scaled,Y, test_size = 0.2,random_state =100)

model  = LogisticRegression()
model.fit(X_train,Y_train)

# Accuracy
print (model.score(X_train, Y_train))

# Save the model
filename = 'trained_model.pkl'
joblib.dump(model,filename)

# Predict 
predictions = model.predict(X_test)
print(predictions)
# Confusion Matrix
cm = metrics.confusion_matrix(Y_test, predictions)
print(cm)

cr = metrics.classification_report(Y_test, predictions)
print (cr)
print(precision_score,recall_score,accuracy_score)