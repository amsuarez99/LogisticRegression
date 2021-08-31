import numpy as np
import pandas as pd
import scipy
import math
from numpy.linalg import norm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


"""
-------------- Utilities ------------------
"""

# rounds up or down
def classifier(predictions):
    classified = []
    for prediction in predictions:
        if prediction >= 0.5:
            classified.append(1)
        else:
            classified.append(0)
    return classified

# returns a vector of p-hat
def predict(X, B):
    predictions = []
    for x in X:
        try:
            ans = math.exp(np.dot(-1 * B.T,x))
        except OverflowError:
            ans = float('inf')
        prediction = 1 / (1 + ans)
        predictions.append(prediction)
    return predictions


# Calculates Gradient of J (in this case logit?)
def gradient(X, y, B):
    n, _ = X.shape
    # Calculate vector mu, which is a vector of predictions (p-hat)
    mu = predict(X, B)
    error = mu - y
    return np.dot(X.T, error)

# Does gradient descent on the model
# X = our dataset
# y = our outputs
# eps = convergence stop
# iterations = iterations lol
# alpha = learning rate
def gradiente_desc(X, y, alpha=0.0001, eps=0.001, iterations = 300):
    n, m = X.shape

    # initialize B0
    B_hist = []
    B = np.ones(m)
    j = 0
    while ((norm(gradient_B := gradient(X,y,B)) > eps) and j < iterations):
        B_hist.append(B)
        B = B - (alpha * gradient_B)
        j+=1
    return B_hist[-1][:]

"""--------------------------------Dataset #1: default.txt--------------------------"""
# Get data of dataset
data = pd.read_csv('data/default.txt', sep='\t')
data = data[['default', 'student', 'balance', 'income']]

# Change 'True' to 1 and 'False' to 0 in student and default columns
data['student'] = data['student'].map(dict(Yes=1, No=0))
data['default']= data['default'].map(dict(Yes=1, No=0))
X = data[['student', 'balance', 'income']].to_numpy()
y = data.default.to_numpy()
X,y

# Instantiate the model
lr = LogisticRegression()

# 0.80 for the training set
# 0.20 for the test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42, shuffle=True)
lr.fit(X_train, y_train)

# Get an array of predicted values
y_prediction = lr.predict(X_test)


B = gradiente_desc(X_train, y_train)
y_pred = np.dot(X_test, B)
classified_predictions = classifier(y_pred)

print("With Sklearn LogisticRegression")
print(confusion_matrix(y_test,y_prediction))
print("accurracy_score: ",accuracy_score(y_test, y_prediction))
print("Our Gradient Descent")
print(confusion_matrix(y_test, classified_predictions))
print("accurracy_score: ",accuracy_score(y_test, classified_predictions))
print("Optimized B Vector: ", B)
# Confustion Matrices Legend
# TRUE NEGATIVE     FALSE POSITIVE
# FALSE NEGATIVE    TRUE POSITIVE


"""--------------------------------Dataset #2: genero.txt--------------------------"""

genero = pd.read_csv('data/genero.txt')
genero['Gender'] = genero['Gender'].map(dict(Male=1, Female=0))

X = genero[['Height','Weight']].to_numpy()
y = genero.Gender.to_numpy()

# Instantiate the model
lr = LogisticRegression()

# 0.80 for the training set
# 0.20 for the test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42, shuffle=True)

lr.fit(X_train, y_train)

# Get an array of predicted values
y_prediction = lr.predict(X_test)

B = gradiente_desc(X_train, y_train)
y_pred = np.dot(X_test, B)
classified_predictions = classifier(y_pred)

print("With Sklearn LogisticRegression")
print(confusion_matrix(y_test,y_prediction))
print("accurracy_score: ",accuracy_score(y_test, y_prediction))
print("Our Gradient Descent")
print(confusion_matrix(y_test, classified_predictions))
print("accurracy_score: ",accuracy_score(y_test, classified_predictions))
print("Optimized B Vector: ", B)
# Confustion Matrices Legend
# TRUE NEGATIVE     FALSE POSITIVE
# FALSE NEGATIVE    TRUE POSITIVE
