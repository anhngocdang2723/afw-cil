import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
#from common.change_rate_data import change_rate_data

def load_data(test_size):
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'Co_Author_250_750.csv'))
    dataset_desc = dataset.describe(include = 'all')
    # ecoli_map = {' im':1.0, ' cp':-1.0, 'imL':-1.0,'imS':-1.0,'imU':-1.0,' om':-1.0,'omL':-1.0,' pp':-1.0}
    # dataset['class'] = dataset['class'].map(ecoli_map)
    X = dataset.drop(['Label'], axis=1)
    y = dataset['Label']
    #Split data
    X_train, X_test, y_train, y_test = tts(X, y, test_size = test_size, random_state = 42, stratify=y)
    #Scalling Data

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    #Analys data
    # pca = PCA(n_components = 15)
    # X_train  = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    return X_train, y_train, X_test, y_test
