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
rf_model = RandomForestRegressor(n_estimators = 20, max_depth = 1000, random_state = 50)
rf_model.fit(x_train, y_train)
features = list(X.columns)

# y_pred = rf_model.predict(x_test)
plt.rcParams['font.size'] = '35'
importance = rf_model.feature_importances_
for each in zip(features,importance):
    print (each) 

plt.figure(figsize=(70,40))     
plt.bar([x[0] for x in zip(features,importance)], importance)
plt.show()
# score  = r2_score(y_test, y_pred)
# print(score)

filename = 'housepricemodel.pkl'
joblib.dump(rf_model,filename)