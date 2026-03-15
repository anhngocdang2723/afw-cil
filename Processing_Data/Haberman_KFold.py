import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

# from common.change_rate_data import change_rate_data

def load_data():
    # [UPDATE]: Fix relative path to work from any working directory
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'haberman.csv')
    dataset = pd.read_csv(dataset_path)
    dataset_desc = dataset.describe(include='all')
    haberman_map = {2: 1.0, 1: -1.0}
    dataset['class'] = dataset['class'].map(haberman_map)
    X = dataset.drop(['class'], axis=1)
    y = dataset['class']
    # y_pos = Counter(y.where(y==1))
    # y_neg = Counter(y.where(y==-1))
    # print(y_pos)
    # print(y_neg)
    X = np.array(X)
    y = np.array(y)
    return X, y

# print(load_data())


