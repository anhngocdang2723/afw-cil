import os
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from common.change_rate_data import change_rate_data

def load_data(test_size,testsize_val):
    # [UPDATE]: Fix relative path to work from any working directory
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'new-thyroid2.csv')
    dataset = pd.read_csv(dataset_path)
    dataset_desc = dataset.describe(include = 'all')
    ecoli1_map = {'positive':1,'negative':-1}
    dataset['class'] = dataset['class'].map(ecoli1_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 5].values


    #Split data
    #X, y = change_rate_data(X, y , new_rate = new_rate)
    X_train, X_test, y_train, y_test = tts(X, y, test_size = test_size, random_state = 42, stratify=y)
    #Scalling Data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    #Analys data
    # pca = PCA(n_components = 15)
    # X_train  = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    X_train_val, X_test_val, y_train_val, y_test_val = tts(X_train, y_train, test_size=testsize_val, random_state=42,stratify=y_train)
    # X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42)
    return X_train_val, y_train_val, X_test_val, y_test_val, X_test, y_test