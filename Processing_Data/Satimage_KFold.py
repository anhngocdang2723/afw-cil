import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from Processing_Data.common.change_rate_data import change_rate_data

def load_data(): 
    # [UPDATE]: Fix relative path to work from any working directory
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'satimage_full.csv')
    dataset = pd.read_csv(dataset_path)
    dataset_desc = dataset.describe(include='all')
    satimage_map = {4: 1, 1: -1, 2: -1, 3: -1, 5: -1, 6: -1, 7: -1}
    dataset['class'] = dataset['class'].map(satimage_map)
    X = dataset.drop(['class'], axis=1)
    y = dataset['class']
    X = np.array(X)
    y = np.array(y)
    return X, y

