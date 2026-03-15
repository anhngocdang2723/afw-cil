import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Processing_Data.common.change_rate_data import change_rate_data
import os

def load_data(test_size,new_rate=1/5):
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'CoAuthor_1800.csv'))
    # diag_map = {-1: -1.0, 1: -1.0}
    # data['Label'] = data['Label'].map(diag_map)
    X = data.values[:, 0:-1]
    y = data.values[:, 7]
    X, y = change_rate_data(X, y , new_rate = new_rate)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return X_train,y_train, X_test, y_test
