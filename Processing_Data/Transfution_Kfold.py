import os
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
#from common.change_rate_data import change_rate_data



def load_data():
    # [UPDATE]: Fix relative path to work from any working directory
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'transfusion.csv')
    dataset = pd.read_csv(dataset_path)
    dataset_desc = dataset.describe(include = 'all')
    transfusion_map = {1:1, 0:-1}
    dataset['whether he/she donated blood in March 2007'] = dataset['whether he/she donated blood in March 2007'].map(transfusion_map)
    X = dataset.drop(['whether he/she donated blood in March 2007'], axis = 1)
    y = dataset['whether he/she donated blood in March 2007']
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, 4].values

    #Split data
    #X, y = change_rate_data(X, y , new_rate = new_rate)
    # X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 42, stratify=y)
    #Scalling Data
    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)
    # return  X_train, X_test, y_train, y_test
    X = np.array(X)
    y = np.array(y)
    return X,y

