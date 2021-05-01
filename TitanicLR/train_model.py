import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# load data to DF
data = pd.read_csv(r'E:\1.DEEPAK Data Science\GITREPO\DMT_MASTER_REPO\TitanicLR\data.csv')
data