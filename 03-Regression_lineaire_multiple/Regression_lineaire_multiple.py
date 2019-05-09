# ------------------------------------------------------
# --------------------- IMPORT -------------------------
# ------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------
# --------------------- DATASET ------------------------
# ------------------------------------------------------

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


# ------------------------------------------------------
# -------------- VARIABLES CATEGORIQUES ----------------
# ------------------------------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])
ohe = OneHotEncoder(categorical_features = [3])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:] 


# ------------------------------------------------------
# ------------------ TRAINING SET ----------------------
# ------------------------------------------------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2)


# ------------------------------------------------------
# ------------ CONSTRUCTION DU MODELE ------------------
# ------------------------------------------------------
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# ------------------------------------------------------
# ------------- NOUVELLES PREDICTIONS ------------------
# ------------------------------------------------------
y_pred = regressor.predict(x_test)
print(y_pred)
y_pred2 = regressor.predict(np.array([[1, 0, 130000, 140000, 300000]])) # Il faut que ça soit un ndarray sinon ça ne marche pas
print('----------------')
print(y_pred2)