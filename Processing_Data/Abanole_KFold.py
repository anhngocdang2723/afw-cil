import os
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from common.change_rate_data import change_rate_data

def load_data():
    # [UPDATE]: Fix relative path to work from any working directory
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'abalone.csv')
    dataset = pd.read_csv(dataset_path)
    dataset_desc = dataset.describe(include = 'all')
    abalone_map = {15 : 1, 1 : -1, 2 : -1, 3 : -1, 4 : -1, 5 : -1, 6 : -1, 7 : -1, 8 : -1, 9 : -1, 10 : -1,
    11 : -1, 12 : -1, 13 : -1, 14 : -1, 16 : -1, 17 : -1, 18 : -1, 19 : -1, 20 : -1, 21 : -1, 22 : -1, 23 : -1,
    24 : -1, 25 : -1, 26 : -1, 27 : -1, 28 : -1, 29 : -1}
    dataset['Rings'] = dataset['Rings'].map(abalone_map)
    X = dataset.drop(['Rings'], axis=1).values
    y = dataset['Rings'].values
    labelencoder_X = LabelEncoder()
    # Encoding the State Categorization
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    onehotencoder_X = OneHotEncoder()
    onehotencoder_X.fit_transform(X).toarray()
    
    X = np.array(X)
    y = np.array(y)
    return X, y