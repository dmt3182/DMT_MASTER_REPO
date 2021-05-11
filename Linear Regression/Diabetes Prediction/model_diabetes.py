import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

data = pd.read_csv(r'E:\1.DEEPAK Data Science\GITREPO\Linear Regression\Diabetes Prediction\diabetes.csv')
print (data.shape)
print (data.info())


x = data.drop(data.Outcome).values
y = data.iloc[:,-1].head(766).values
print(x.shape)


scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)
print (scaled_x.shape,y.shape)
print (data.info())
# Train Test Spit

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 20)
model = LogisticRegression()
model.fit(x_train, y_train)
score = model.score(x_train, y_train)
print (score)


cv = KFold(n_splits=2, random_state=100)
model = LogisticRegression()
model.fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train, cv=cv,scoring = 'accuracy')
print ("KFOLD :", np.mean(scores), np.std(scores))

cv = RepeatedKFold(n_splits=10,n_repeats=3, random_state=1)
model1 = LogisticRegression()
scores1 = cross_val_score(model1, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
print ("KFOLDREPEATED :", np.mean(scores1), np.std(scores1))

ypred = model.predict(x_test)

mae = metrics.mean_absolute_error(y, ypred)
mse = metrics.mean_squared_error(y, ypred)
rmse = np.sqrt(mse)

print (mae,mse,rmse)