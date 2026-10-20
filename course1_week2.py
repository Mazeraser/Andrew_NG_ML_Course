#linear regression with multiple features
import random

import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import random

def compute_function(x, w, b):
    return np.dot(x,w)+b
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost=0
    for i in range(m):
        f_wb_i = compute_function(X[i],w,b)
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost/(m*2)
    return cost
def compute_gradient(X, y, w, b):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        err = compute_function(X[i],w,b)-y[i]
        for j in range(n):
            dj_dw[j] += err*X[i,j]
        dj_db += err
    return dj_dw/m, dj_db/m
def gradient_descent(X,y, start_w, start_b, alpha, iterations):
    w = copy.deepcopy(start_w)
    b = start_b
    J_history = []

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(compute_cost(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")


    return w,b, J_history

#normalization
def feature_scaling(X):
    x_maxs = X.max(axis=0)
    normalized_X = (X) / x_maxs

    return normalized_X
def mean_normalization(X):
    averages = X.mean(axis=0)
    x_mins = X.min(axis=0)
    x_maxs = X.max(axis=0)

    ranges = x_maxs - x_mins
    ranges[ranges == 0] = 1

    normalized_X = (X - averages) / ranges
    return normalized_X
def Z_score_normalization(X):
    averages = X.mean(axis=0)
    deviations = X.std(axis=0)

    normalized_X = (X - averages) / deviations
    return normalized_X

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 10000*random.random()
w_init = np.array([ 3*random.random(), 50*random.random(), -100*random.random(), -40*random.random() ])
alpha = 5e-1
iterations = 20

X_feature_scaled = feature_scaling(X_train)
X_mean_normalization = mean_normalization(X_train)
X_Z_score = Z_score_normalization(X_train)
print("-"*50)
print(f"---Normalized features--- "
      f"\n feature scaling:\n{X_feature_scaled} "
      f"\n mean normalization:\n{X_mean_normalization} "
      f"\n Z-score normalization:\n{X_Z_score} ")
print("-"*50)

print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")
print(f"input X: \n{X_train}")
print(f"input Y: \n{y_train}")
print("-"*50)
print("---Z-score normalized features---")
w_final, b_final, J_hist = gradient_descent(X_Z_score, y_train, w_init, b_init, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
print("-"*50)

y_pred = X_Z_score @ w_final + b_final   # shape (3,), предсказания для всех примеров

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for i in range(X_train.shape[1]):
    ax[i].scatter(X_train[:, i], y_train, color='red', label='true')
    ax[i].scatter(X_train[:, i], y_pred, color='blue', marker='x', label='pred')
    ax[i].set_xlabel(f"x{i+1}")
    ax[i].set_ylabel("y")
    ax[i].legend()
fig.suptitle("True vs predicted (per feature)")
fig.tight_layout()
plt.show()
plt.plot(J_hist)
plt.xlabel("iterations")
plt.ylabel("J_history")
plt.show()