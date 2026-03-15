import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os


def load_data(test_size):
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'ecoli_new.csv'))
    dataset_desc = dataset.describe(include = 'all')
    ecoli_map = {' im':1.0, ' cp':-1.0, 'imL':-1.0,'imS':-1.0,'imU':-1.0,' om':-1.0,'omL':-1.0,' pp':-1.0}
    dataset['class'] = dataset['class'].map(ecoli_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 7].values


    # labelencoder_X = LabelEncoder()
    # # Encoding the Sex Categorization
    # X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    # onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
    # onehotencoder_X.fit_transform(X).toarray()

    #Split data
    X_train, X_test, y_train, y_test = tts(X, y, test_size = test_size, random_state = 42, stratify=y)
    #Scalling Data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    #Analys data
    # pca = PCA(n_components = 15)
    # X_train  = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    return X_train, y_train, X_test, y_test

