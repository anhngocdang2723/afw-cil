import os
from tkinter.tix import X_REGION
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from common.change_rate_data import change_rate_data

def load_data():
    # [UPDATE]: Fix absolute Windows path to work from any working directory
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'ecoli_new.csv')
    dataset = pd.read_csv(dataset_path)
    dataset_desc = dataset.describe(include = 'all')
    ecoli_map = {' im':1.0, ' cp':-1.0, 'imL':-1.0,'imS':-1.0,'imU':-1.0,' om':-1.0,'omL':-1.0,' pp':-1.0}
    dataset['class'] = dataset['class'].map(ecoli_map)
    X = dataset.drop(['class'], axis=1)
    y = dataset['class']
    X = np.array(X)
    y = np.array(y)
    return X, y

# print(load_data())