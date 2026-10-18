#linear regression with cost function and gradient descent

import numpy as np
import matplotlib.pyplot as plt
import math
import random

def compute_function(x, w, b):
    return x*w+b
def compute_cost(x, y, w, b):
    m = x.shape[0]
    f_wb = compute_function(x, w, b)
    return 1/(2*m)*(np.sum( (f_wb-y)**2 ))
def compute_gradient(x, y, w, b):
    error = compute_function(x,w,b) - y
    dj_dw = np.mean(error * x)
    dj_db = np.mean(error)
    #np.mean - среднее арифметическое

    return dj_dw, dj_db
def gradient_descent(x,y, start_w, start_b, alpha, iterations):
    w = start_w
    b = start_b
    J_history = []
    P_history = []

    for i in range(iterations):
        derivative = compute_gradient(x, y, w, b)
        w = w - alpha*derivative[0]
        b = b - alpha*derivative[1]
        J_history.append(compute_cost(x,y,w,b))
        P_history.append([w,b])
        if(i%(math.ceil(iterations/10))==0):
            print(f"Iterations: {i}, dj_dw: {derivative[0]: 0.2e}, dj_db: {derivative[1]: 0.2e}, "
                  +f"Current J:{compute_cost(x,y,w,b): 0.4e}, w:{w: 0.2e}, b:{b: 0.2e}")

    return w,b, J_history, P_history

x_train = np.array([i*10 for i in range(10)])
y_train = 5*x_train + 8 + 1e-1*random.random()
w = 0
b = 0
alpha = 1e-4
iterations = 100000

w, b, J_history, P_history = gradient_descent(x_train, y_train, w, b, alpha, iterations)

plt.scatter(x=x_train, y=y_train, color='red', label='Training Data')
plt.plot(x_train, compute_function(x_train, w, b), color='blue', linewidth=2, label="f_wb")
plt.title("Linear regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.plot(J_history)
plt.title("Cost convergence")
plt.xlabel("iterations")
plt.ylabel("J(w,b)")
plt.show()