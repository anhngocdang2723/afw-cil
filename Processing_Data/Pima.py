import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# from common.change_rate_data import change_rate_data

def load_data():
    # [UPDATE]: Fix relative path to work from any working directory
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'diabetes.csv')
    dataset = pd.read_csv(dataset_path)
    dataset_desc = dataset.describe(include='all')
    pimaIndians_map = {1: 1, 0: -1}
    dataset['Outcome'] = dataset['Outcome'].map(pimaIndians_map)
    X = dataset.drop(['Outcome'], axis=1)
    y = dataset['Outcome']
    X = np.array(X)
    y = np.array(y)
    return X,y




