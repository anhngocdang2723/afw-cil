import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import TomekLinks 
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from data.common.change_rate_data import change_rate_data


def load_data(test_size, new_rate):
 #load data
    data = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/FUZZY SVM/CODE/07_04_2022/fuzzy_svm/data/indian_liver_patient.csv')
    Gender_map = {'Female': 0, 'Male': 1.0}
    #convert string to numberic
    data['Gender'] = data['Gender'].map(Gender_map)
    Dataset_map = {1 : -1, 2: 1}
    data['Dataset'] = data['Dataset'].map(Dataset_map)
    #Define X, y 
    y = data['Dataset']
    X = data.iloc[:, 1:10]
    #Scaler data
    X = X.to_numpy()
    y = y.to_numpy()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X[:,2:10] = imputer.fit_transform(X[ :,2:10])
    # Analysis the data
    ## [UPDATE - Start]: Fix lỗi bug toarray của OneHotEncoder và logic dán đè X
    ## Giống như bên file churn, sử dụng ColumnTransformer để giữ lại các cột số 
    ## Thay vì gọi OneHotEncoder.fit_transform(X).toarray() sai cách 
    from sklearn.compose import ColumnTransformer
    
    # Cột 1 (sau khi slice data.iloc[:, 1:10]) chính là cột Gender cần OneHot Encode
    labelencoder_X = LabelEncoder()
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    
    ct = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [1])
        ],
        remainder='passthrough'
    )
    X = ct.fit_transform(X)
    ## [UPDATE - End] --- IGNORE ---

    X, y = change_rate_data(X, y , new_rate = new_rate)
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)

    # X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.5, random_state = 1)
    
    #Sclaler data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    pca = PCA(n_components = None)
    X_train  = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)   
    pca = PCA(n_components = 6)
    X_train  = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    #split data and label
    return X_train, y_train, X_test, y_test


# def load_data(test_size):
#     #load data
#     data = pd.read_csv('data/indian_liver_patient.csv')
#     Gender_map = {'Female': 0, 'Male': 1.0}
#     #convert string to numberic
#     data['Gender'] = data['Gender'].map(Gender_map)
#     Dataset_map = {1 : -1.0, 2: 1.0}
#     data['Dataset'] = data['Dataset'].map(Dataset_map)
#     #Define X, y 
#     y = data['Dataset']
#     X = data.iloc[:, 1:10]
#     #Scaler data
#     X = X.to_numpy()
#     y = y.to_numpy()
#     imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#     X[:,2:10] = imputer.fit_transform(X[ :,2:10])
#     X_train, X_test, y_train, y_test = tts(X, y, test_size = test_size, random_state = 42)
#     #Sclaler data
#     sc_X = StandardScaler()
#     X_train = sc_X.fit_transform(X_train)
#     X_test = sc_X.transform(X_test)
#     # Analysis the data

#     pca = PCA(n_components = 9)
#     X_train  = pca.fit_transform(X_train)
#     X_test = pca.transform(X_test)
    
#     #split data and label
#     return X_train, y_train, X_test, y_test


# =============================================================================
# 
# data = pd.read_csv('E:/My project/soict/cvxopt_2/data/indian_liver_patient.csv')
# Gender_map = {'Female': 1.0, 'Male': 0.0}
# #convert string to numberic
# data['Gender'] = data['Gender'].map(Gender_map)
# Dataset_map = {1 : -1, 2: 1}
# data['Dataset'] = data['Dataset'].map(Dataset_map)
# #Define X, y 
# y = data['Dataset']
# X = data.iloc[:, 1:10]
# #Scaler data
# X_normalized = MinMaxScaler().fit_transform(X.values)
# X = pd.DataFrame(X_normalized)
# X = X.to_numpy()
# y = y.to_numpy()
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# X[:,2:10] = imputer.fit_transform(X[ :,2:10])
# =============================================================================
