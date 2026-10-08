import numpy as np
import matplotlib.pyplot as plt

def compute_function(x, w, b):
    return x*w+b
def compute_cost(x, y, w, b):
    m = x.shape[0]
    f_wb = compute_function(x, w, b)
    return 1/(2*m)*(np.sum( (f_wb-y)**2 ))

x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([10, 4, 5, 8, 13])
w = 2
b = 2

print(f"Текущая затрата:  {compute_cost(x_train, y_train, w, b)}")

plt.scatter(x=x_train, y=y_train, color='red', label='Training Data')
plt.plot(x_train, compute_function(x_train, w, b), color='blue', linewidth=2, label="f_wb")
plt.title("Linear regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
