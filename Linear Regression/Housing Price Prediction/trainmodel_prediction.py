import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import joblib

# READ DATA 
data = pd.read_csv(r'E:\1.DEEPAK Data Science\GITREPO\Linear Regression\Housing Price Prediction\train.csv')
x = data.drop(columns=['SalePrice'],axis = 1)
y = data['SalePrice']

# imputing the data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputed_Data = pd.DataFrame(imputer.fit_transform(x), columns = x.columns)

# print (imputed_Data)

columns_to_encode = x.select_dtypes(include ='object').columns
columns_to_scale  = x.select_dtypes(include = ['int64','float64']).columns


# Instantiate encoder/scaler
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)

# Scale and Encode Separate Columns
encoded_columns = ohe.fit_transform(imputed_Data[columns_to_encode])
scaled_columns  = scaler.fit_transform(imputed_Data[columns_to_scale]) 

# Concatenate (Column-Bind) Processed Columns Back Together
processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1)
print(processed_data)



# Feature Selection 


# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(processed_data,y, test_size = 0.2,random_state = 42)

# Linear Model
model = linear_model.LinearRegression()
reg = model.fit(x_train,y_train)
r_sq = reg.score(x_train,y_train)
print("Coiefficient of determination : " ,  r_sq)

print(model.coef_)

# # save the model to disk
# filename = 'finalmodel_prediction.pkl'
# joblib.dump(model,filename)