import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data.common.change_rate_data import change_rate_data

def load_data(test_size,new_rate):
    dataset = pd.read_csv('./data/datasets/churn.csv')
    dataset_desc = dataset.describe(include = 'all')
    Churn_map = {'False.' : -1, 'True.': 1}
    dataset['Churn?'] = dataset['Churn?'].map(Churn_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 20].values
    
    ## [UPDATE - Start]: Sửa lỗi mất dữ liệu khi dùng OneHotEncoder
    ## Logic cũ ghi đè lên biến X gây mất các đặc trưng khác. 
    ## Thay bằng ColumnTransformer để gom cụm mã hóa các cột phân loại (0, 3, 4, 5, 6) 
    ## và giữ nguyên các cột số khác với remainder='passthrough'.
    ct = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(sparse_output=False), [0, 3, 4, 5, 6])
        ],
        remainder='passthrough'
    )
    X = ct.fit_transform(X)
    ## [UPDATE - End]
    
    #Split data
    X, y = change_rate_data(X, y , new_rate = new_rate)
    X_train, X_test, y_train, y_test = tts(X, y, test_size = test_size, random_state = 42, stratify=y)
    #Scalling Data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    #Analys data
    pca = PCA(n_components = 15)
    X_train  = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, y_train, X_test, y_test