# ------------------------------------------------------
# --------------------- IMPORT -------------------------
# ------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------
# --------------------- DATASET ------------------------
# ------------------------------------------------------
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values 
Y = dataset.iloc[:, -1].values 


# ------------------------------------------------------
# ------------------ TRAINING SET ----------------------
# ------------------------------------------------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25) 


# ------------------------------------------------------
# ---------------- FEATURE SCALING----------------------
# ------------------------------------------------------
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# ------------------------------------------------------
# ------------- CONSTRUCTION DU MODELE -----------------
# ------------------------------------------------------
from sklearn.svm import SVC 

classifier = SVC(kernel = 'linear')
classifier.fit(x_train, y_train)

# ------------------------------------------------------
# ------------------- PREDICTION  ----------------------
# ------------------------------------------------------
y_pred = classifier.predict(x_test)
#print(y_pred)


# ------------------------------------------------------
# -------------- MATRICE DE CONFUSION ------------------
# ------------------------------------------------------
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

# ------------------------------------------------------
# ----------------- VISUALISATION ----------------------
# ------------------------------------------------------
from matplotlib.colors import ListedColormap

X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.4, cmap = ListedColormap(('red', 'green')))
#alpha c'est juste pour le contraste
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Résultats du Training set')
plt.xlabel('Age')
plt.ylabel('Salaire Estimé')
plt.legend()
plt.show()