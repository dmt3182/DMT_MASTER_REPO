import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv(r'E:\1.DEEPAK Data Science\GITREPO\DMT_MASTER_REPO\RandomForest\Salary.csv')
print(data.info())

X = data['YearsExperience'].values.reshape(-1,1)
Y = data['Salary']

X_train, X_test, Y_train,Y_test = train_test_split( X,Y,test_size = 0.2,random_state = 42 )
regressor = RandomForestRegressor(n_estimators=50, max_depth = 3,random_state= 42)
regressor.fit(X_train, Y_train)



filename = 'savemodel.pkl'
joblib.dump(regressor, filename)