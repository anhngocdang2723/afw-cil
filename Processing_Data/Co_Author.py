import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
#from common.change_rate_data import change_rate_data

def load_data():
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', 'CoAuthor_100_500.csv'))
    dataset_desc = dataset.describe(include = 'all')
    # ecoli_map = {' im':1.0, ' cp':-1.0, 'imL':-1.0,'imS':-1.0,'imU':-1.0,' om':-1.0,'omL':-1.0,' pp':-1.0}
    # dataset['class'] = dataset['class'].map(ecoli_map)
    X = dataset.drop(['Label'], axis=1)
    y = dataset['Label']
    return X, y



# print(load_data())