import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# from common.change_rate_data import change_rate_data

def load_data():
    # [UPDATE]: Fix absolute Windows path to work from any working directory
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'yeast.csv')
    dataset = pd.read_csv(dataset_path)
    dataset_desc = dataset.describe(include='all')
    yeast_map = {'ME2': 1, 'CYT': -1, 'ERL': -1, 'EXC': -1, 'ME1': -1, 'ME3': -1, 'MIT': -1, 'NUC': -1, 'POX': -1, 'VAC': -1}
    dataset['name'] = dataset['name'].map(yeast_map)
    X = dataset.drop(['name'], axis=1)
    y = dataset['name']
    X = np.array(X)
    y = np.array(y)
    return X, y






