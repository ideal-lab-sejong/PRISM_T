import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "Classical")


def list_datasets(data_dir=DATA_DIR):
    return sorted(f[:-4] for f in os.listdir(data_dir) if f.endswith(".npz"))


def load(dataset_name, seed, data_dir=DATA_DIR, test_size=0.5, scaler="standard"):
    data = np.load(os.path.join(data_dir, dataset_name + ".npz"), allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=int).ravel()

    X_normal, X_anomaly = X[y == 0], X[y == 1]
    X_train, X_test_normal = train_test_split(X_normal, test_size=test_size, random_state=seed)
    X_test = np.vstack([X_test_normal, X_anomaly])
    y_test = np.concatenate([np.zeros(len(X_test_normal), int), np.ones(len(X_anomaly), int)])

    scl = MinMaxScaler() if scaler == "minmax" else StandardScaler()
    scl.fit(X_train)
    X_train, X_test = scl.transform(X_train), scl.transform(X_test)

    for A in (X_train, X_test):
        cm = np.nanmean(A, axis=0); cm[np.isnan(cm)] = 0.0
        ii = np.where(np.isnan(A)); A[ii] = np.take(cm, ii[1])
    return X_train, X_test, y_test
