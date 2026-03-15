import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler

def load_data(test_size, testsize_val):
    # Follow convention of other Processing_Data modules: load from Processing_Data/dataset/
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'abalone19.csv')
    dataset = pd.read_csv(dataset_path)

    # Map Class: positive -> 1, negative -> -1
    class_map = {'positive': 1.0, 'negative': -1.0}
    dataset['Class'] = dataset['Class'].map(class_map)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    y = np.asarray(y, dtype=float)

    # Encode Sex column (M, F, I) with LabelEncoder
    labelencoder = LabelEncoder()
    X[:, 0] = labelencoder.fit_transform(X[:, 0])

    # Convert all X values to float
    X = X.astype(float)

    # Split into train and test
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale features
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # Further split train into train_val and test_val
    X_train_val, X_test_val, y_train_val, y_test_val = tts(
        X_train, y_train, test_size=testsize_val, random_state=42, stratify=y_train
    )

    return X_train_val, y_train_val, X_test_val, y_test_val, X_test, y_test
