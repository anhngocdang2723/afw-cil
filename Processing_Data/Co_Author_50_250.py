import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Processing_Data.common.change_rate_data import change_rate_data
import os

# def load_data(new_rate):
#     dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'Co_Author_Static_50_250.csv'))
#     dataset_desc = dataset.describe(include = 'all')
#     # ecoli_map = {' im':1.0, ' cp':-1.0, 'imL':-1.0,'imS':-1.0,'imU':-1.0,' om':-1.0,'omL':-1.0,' pp':-1.0}
#     # dataset['class'] = dataset['class'].map(ecoli_map)
#     # X = dataset.drop(['Label'], axis=1)
#     # y = dataset['Label']
#     X = dataset.values[:, 0:-1]
#     y = dataset.values[:, 7]
#     print(y)
#     X, y = change_rate_data(X, y , new_rate = new_rate)
#     return X, y

# def load_data():
#     dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'Co_Author_50_250.csv'))
#     dataset_desc = dataset.describe(include = 'all')
#     # ecoli_map = {' im':1.0, ' cp':-1.0, 'imL':-1.0,'imS':-1.0,'imU':-1.0,' om':-1.0,'omL':-1.0,' pp':-1.0}
#     # dataset['class'] = dataset['class'].map(ecoli_map)
#     # X = dataset.drop(['Label'], axis=1)
#     # y = dataset['Label']
#     X = dataset.values[:, 0:-1]
#     y = dataset.values[:, 7]
#     print(y)
#     return X, y

def load_data(test_size,new_rate=1/5):
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'Co_Author_50_250.csv'))
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




# print(load_data())