import os
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from common.change_rate_data import change_rate_data

# [UPDATE]: Wrap global-level code in if __name__ == '__main__' to prevent execution at import time
if __name__ == '__main__':
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'page-blocks.csv'))
    dataset_desc = dataset.describe(include = 'all')
    pageblocks_map = {5 : 1, 1: -1, 2: -1, 3: -1, 4: -1}
    dataset['class'] = dataset['class'].map( pageblocks_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 10].values

    # dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'abalone.csv'))
    # dataset_desc = dataset.describe(include = 'all')
    # abalone_map = {15 : 1, 1 : -1, 2 : -1, 3 : -1, 4 : -1, 5 : -1, 6 : -1, 7 : -1, 8 : -1, 9 : -1, 10 : -1,
    # 11 : -1, 12 : -1, 13 : -1, 14 : -1, 16 : -1, 17 : -1, 18 : -1, 19 : -1, 20 : -1, 21 : -1, 22 : -1, 23 : -1,
    # 24 : -1, 25 : -1, 26 : -1, 27 : -1, 28 : -1, 29 : -1}
    # dataset['Rings'] = dataset['Rings'].map(abalone_map)
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 8].values

    # labelencoder_X = LabelEncoder()
    # # Encoding the Sex Categorization
    # X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    # onehotencoder_X = OneHotEncoder()
    # onehotencoder_X.fit_transform(X).toarray()

    # print(X[:, 0])
    # print(y)

    # dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'yeast.csv'))
    # dataset_desc = dataset.describe(include = 'all')
    # yeast_map = {'ME2':1, 'CYT':-1, 'ERL':-1,'EXC':-1,'ME1':-1,'ME3':-1,'MIT':-1,'NUC':-1,'POX':-1,'VAC':-1}
    # dataset['name'] = dataset['name'].map(yeast_map)
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 8].values

    # dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'satimage_full.csv'))
    # dataset_desc = dataset.describe(include = 'all')
    # satimage_map = {4:1, 1:-1, 2:-1, 3:-1, 5:-1, 6:-1, 7:-1}
    # dataset['class'] = dataset['class'].map(satimage_map)
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 36].values


    # dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'ecoli.csv'))
    # dataset_desc = dataset.describe(include = 'all')
    # ecoli_map = {' im':1,' cp':-1,'imL':-1,'imS':-1,'imU':-1,' om':-1,'omL':-1,' pp':-1}
    # dataset['class'] = dataset['class'].map(ecoli_map)
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 8].values

    # labelencoder_X = LabelEncoder()
    # # Encoding the Sex Categorization
    # X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    # onehotencoder_X = OneHotEncoder()
    # onehotencoder_X.fit_transform(X).toarray()

    # print(X[:,0])


    # dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'transfusion.csv'))
    # dataset_desc = dataset.describe(include = 'all')
    # transfusion_map = {1:1, 0:-1}
    # dataset['whether he/she donated blood in March 2007'] = dataset['whether he/she donated blood in March 2007'].map(transfusion_map)
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 4].values

    # dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'haberman.csv'))
    # dataset_desc = dataset.describe(include = 'all')
    # haberman_map = {2:1, 1:-1}
    # dataset['class'] = dataset['class'].map(haberman_map)
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 3].values

    # dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'waveform.csv'))
    # dataset_desc = dataset.describe(include = 'all')
    # waveform_map = {0:1, 1:-1, 2:-1}
    # dataset['class'] = dataset['class'].map(waveform_map)
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 21].values


    # dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'diabetes.csv'))
    # dataset_desc = dataset.describe(include = 'all')
    # pimaIndians_map = {1:1, 0:-1}
    # dataset['Outcome'] = dataset['Outcome'].map(pimaIndians_map)
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 8].values

    # dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'shuttle-c2-vs-c4.csv'))
    # dataset_desc = dataset.describe(include = 'all')
    # shuttle_map = {'positive':1,'negative':-1}
    # dataset['class'] = dataset['class'].map(shuttle_map)
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 9].values

    # X, y = change_rate_data(X, y , new_rate = 1/7)

    # print(X)
    # print(y)

def load_data():
    # [UPDATE]: Fix relative path to work from any working directory
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'DataTest.csv')
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(['class'], axis=1)
    X = X.values
    y = dataset['class']
    return X, y

