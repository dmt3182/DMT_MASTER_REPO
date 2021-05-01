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
X = data['YearsExperience'].values.reshape(-1,1)
Y = data['Salary']

X_train, X_test, Y_train,Y_test = train_test_split( X,Y,test_size = 0.2,random_state = 42 )

joblib_model = joblib.load('savemodel.pkl')

# # Predict
y_pred = joblib_model.predict(X_test)
print (y_pred)

s = joblib_model.score(Y_test.values.reshape(-1,1),y_pred.reshape(-1,1))
print ('Score :',s)

df=pd.DataFrame({'Actual':Y_test, 'Predicted':y_pred})
print (df)

print('MAE :' , mean_absolute_error(Y_test,y_pred))

print('MSE :' , mean_squared_error(Y_test,y_pred))

print('R2 :' , r2_score(Y_test,y_pred))

# Feature importance

importance = joblib_model.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()



