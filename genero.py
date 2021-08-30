import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm

def gradient(X, y, B):
  n, _ = X.shape
  prediction = np.dot(X, B)
  error = prediction - y
  return np.dot(X.T, error) * (2 / n)

def gradiente_desc(X, y, alpha=0.0001, eps=0.001, iterations = 300):
  n, m = X.shape

  # X lives in space n*(m+1) to account for the intercept
  aux = np.ones((n, 1))
  X = np.concatenate((aux, X), axis=1)

  # initialize B0
  B_hist = []
  B = np.ones(m + 1)
  j = 0
  while ((norm(gradient_B := gradient(X,y,B)) > eps) and j < iterations):
      B_hist.append(B)
      B = B - (alpha * gradient_B)
      j+=1
      # print('( iteration:', j, ')', 'B norm:', norm(gradient_B))
  return B_hist[-1][:]

# Setup
data = pd.read_csv('genero.txt')
data = data[['Height', 'Weight']]

X = data.Height.to_numpy()
X = X.reshape(-1, 1)
y = data.Weight.to_numpy()

# Regresión lineal
# Aplicar la función LinearRegression de Scikit-Learn para generar el modelo
lr = LinearRegression()

# 0.80 para el training
# 0.20 para el test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42, shuffle=True) # random_state is for shuffling
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Evaluación
print('---- Sklearn Linear Regression ----')
print('weight = %.2fheight + %.2f' % (lr.coef_[0], lr.intercept_))

print('Mean Squared Error: %.2f'
      % mean_squared_error(y_test, y_pred))
print('R2 Score: %.2f' % r2_score(y_test, y_pred))

# ---------------------------------------------------------------
# Regresión Lineal
B = gradiente_desc(X_train, y_train)
y_pred = X_test.dot(B[1]) + B[0]

print('---- Our Gradient Descent ----')
print('weight = %.2fheight + %.2f' % (B[1], B[0]))

print('Mean Squared Error: %.2f'
      % mean_squared_error(y_test, y_pred))
print('R2 Score: %.2f' % r2_score(y_test, y_pred))
