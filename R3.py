import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

data = pd.read_csv("Concrete Compressive Strength.csv")
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title("Matrice de corrélation")
plt.show(block=True)

data = data.drop(["coarseagg","slag","fineagg","ash","age_strength_proxy"], axis=1)

X = data[[ "superplastic","water_cement_ratio", 
          "total_binder", "aggregate_to_cement", "cement_water_interaction","age","cement"]]
y = data["strength"]

def standardize(X):
    return (X - X.mean()) / np.std(X, axis=0)

X = np.array(standardize(X))
y = np.array(y)  # shape (1030,)

m = len(y)

def predict(X, w, b):
    return X @ w + b

def compute_loss(y, y_pred):
    return np.mean((y - y_pred)**2)

def gradient_descent(X, y, w, b, a, N):
    losses = []
    for i in range(N):
        y_pred = predict(X, w, b)
        dw = (-2/m) * X.T @ (y - y_pred)
        db = (-2/m) * np.sum(y - y_pred)
        w = w - a * dw
        b = b - a * db
        losses.append(compute_loss(y, y_pred))
    return w, b, losses

w = np.zeros(7)
b = 0
a = 0.2
N = 5000

w, b, losses = gradient_descent(X, y, w, b, a, N)
y_pred = predict(X, w, b)

print("MSE:", compute_loss(y, y_pred))

def r2_score(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - (ss_res / ss_tot)

print("R²:", r2_score(y, y_pred))

# Courbe de convergence
plt.plot(range(N), losses)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Courbe de convergence")
plt.show(block=True)

# Réel vs Prédit
plt.plot(y_pred,label="valeur prédites")
plt.plot(y,label="valeurs réelles")
plt.xlabel("numéro de l'échantillon")
plt.ylabel("résistance")
plt.show()