import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.under_sampling import TomekLinks 
# from ..common.change_rate_data import  change_rate_data
from data.common.change_rate_data import change_rate_data
def load_data(epoch_tl,test_size, new_rate):
    data = pd.read_csv('./data/datasets/CoAuthor_100_500.csv')
    diag_map = {-1: -1.0, 1: 1.0}
    data['Label class'] = data['Label class'].map(diag_map)
    X = data.values[:, 0:-1]
    y = data.values[:, 7]
    # X = data.drop(['Lable class'], axis=1).values
    # y = data['Label class'].values
    X, y = change_rate_data(X, y , new_rate = new_rate)
    tl = TomekLinks(sampling_strategy = 'not minority')
    print('Original dataset shape %s' % Counter(y))
    for i in range(0,epoch_tl):
        X, y = tl.fit_resample(X, y)[:2] ## trả về X, y sau khi áp dụng TomekLinks
        print('After TomekLink dataset shape %s' % Counter(y))
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    # X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42)
    return X_train,y_train, X_test, y_test
# X_train,y_train, X_test, y_test = load_data(test_size =0.5)
# print(X_train.shape)
# print(X_test.shape)