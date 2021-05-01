import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import joblib   

data = pd.read_csv(r'E:\1.DEEPAK Data Science\GITREPO\DMT_MASTER_REPO\housesales\kc_house_data.csv')
X = data.drop(['price','id','date'], axis = 1)
Y = data.price
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3,random_state = 24)

model = joblib.load('housepricemodel.pkl')
y_pred = model.predict(x_test)
y_pred = y_pred.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)
score1 = model.score(y_test, y_pred)
print (score1)

