# ------------------------------------------------------
# --------------------- IMPORT -------------------------
# ------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------
# --------------------- DATASET ------------------------
# ------------------------------------------------------
dataset = pd.read_csv('Data.csv') 

X = dataset.iloc[:, :-1].values 
Y = dataset.iloc[:, -1].values


# ------------------------------------------------------
# ----------- GERER LES DONNEES MANQUANTES -------------
# ------------------------------------------------------
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X[:, 1:3])
X[:, 1:3] = imp.transform(X[:, 1:3]) 


# ------------------------------------------------------
# -------------- VARIABLES CATEGORIQUES ----------------
# ------------------------------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()

X[:, 0] = le.fit_transform(X[:, 0]) 
ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()
Y = le.fit_transform(Y)


# ------------------------------------------------------
# ------------------ TRAINING SET ----------------------
# ------------------------------------------------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2) 

# ------------------------------------------------------
# ---------------- FEATURE SCALING----------------------
# ------------------------------------------------------
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
