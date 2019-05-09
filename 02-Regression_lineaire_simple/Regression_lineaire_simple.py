# ------------------------------------------------------
# --------------------- IMPORT -------------------------
# ------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------
# --------------------- DATASET ------------------------
# ------------------------------------------------------
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1]


# ------------------------------------------------------
# ------------------ TRAINING SET ----------------------
# ------------------------------------------------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = float(1/3))


# ------------------------------------------------------
# ------------- CONSTRUCTION DU MODELE  ----------------
# ------------------------------------------------------
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)


# ------------------------------------------------------
# ------------------- PREDICTION  ----------------------
# ------------------------------------------------------
y_pred = regressor.predict(x_test)
valeur = regressor.predict([[15]])
#print(valeur)


# ------------------------------------------------------
# ------------------ VISUALISATION  --------------------
# ------------------------------------------------------
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, y_pred, color = 'blue')
plt.title('Salaire vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salaire') 
plt.show()