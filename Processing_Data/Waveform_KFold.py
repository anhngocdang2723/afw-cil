import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Processing_Data.common.change_rate_data import change_rate_data
import os

# from common.change_rate_data import change_rate_data

def load_data(new_rate): #1/5, 1/7, 1/9, 1/11, 1/15
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'waveform.csv'))
    dataset_desc = dataset.describe(include='all')
    waveform_map = {0:1, 1:-1, 2:-1}
    dataset['class'] = dataset['class'].map( waveform_map)
    X = dataset.drop(['class'], axis=1)
    y = dataset['class']
    X, y = change_rate_data(X, y , new_rate = new_rate)
    X = np.array(X)
    y = np.array(y)
    return X, y






