import pandas as pd
import numpy as np
import joblib
# load the model
filename = 'finalmodel_prediction.pkl'
loaded_model = joblib.load(filename)

# live data