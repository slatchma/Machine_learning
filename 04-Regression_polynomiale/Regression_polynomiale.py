# ------------------------------------------------------
# --------------------- IMPORT -------------------------
# ------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------
# --------------------- DATASET ------------------------
# ------------------------------------------------------
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values 
Y = dataset.iloc[:, -1].values 


# ------------------------------------------------------
# ------------- CONSTRUCTION DU MODELE  ----------------
# ------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4) 
X_poly = poly_reg.fit_transform(X) 
regressor = LinearRegression()
regressor.fit(X_poly, Y)


# ------------------------------------------------------
# ------------------- PREDICTION  ----------------------
# ------------------------------------------------------
valeur = regressor.predict([[15]])
#print(valeur)


# ------------------------------------------------------
# ------------ VISUALISER LES RESULTATS-----------------
# ------------------------------------------------------
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X_poly), color = 'blue')
plt.title('Salaire vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()

# Visualiser les résultats (courbe plus lisse)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Salaire vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()