import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
# from Processing_Data.common.change_rate_data import change_rate_data

# def load_data(new_rate):
#     dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'Co_Author_Static_50_250.csv'))
#     dataset_desc = dataset.describe(include = 'all')
#     # ecoli_map = {' im':1.0, ' cp':-1.0, 'imL':-1.0,'imS':-1.0,'imU':-1.0,' om':-1.0,'omL':-1.0,' pp':-1.0}
#     # dataset['class'] = dataset['class'].map(ecoli_map)
#     # X = dataset.drop(['Label'], axis=1)
#     # y = dataset['Label']s
#     X = dataset.values[:, 0:-1]
#     y = dataset.values[:, 7]
#     print(y)
#     X, y = change_rate_data(X, y , new_rate = new_rate)
#     return X, y

def load_data():
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'Co_Author_300_1500_4.csv'))
    dataset_desc = dataset.describe(include = 'all')
    # ecoli_map = {' im':1.0, ' cp':-1.0, 'imL':-1.0,'imS':-1.0,'imU':-1.0,' om':-1.0,'omL':-1.0,' pp':-1.0}
    # dataset['class'] = dataset['class'].map(ecoli_map)
    # X = dataset.drop(['Label'], axis=1)
    # y = dataset['Label']
    X = dataset.values[:, 0:-1]
    y = dataset.values[:, 7]
    print(y)
    return X, y

load_data()



# print(load_data())