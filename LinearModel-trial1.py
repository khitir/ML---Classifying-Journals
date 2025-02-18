import numpy as np
import math

# Function to calculate m and b
def linear_regression(X, y):
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    m = numerator / denominator
    b = y_mean - (m * x_mean)
    return m, b
 
# Function to calculate prediction
def predict(X, m, b):
    return m * X + b
 
# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
 
col start="3"

 
<pre class="lang:default decode:true">X = np.array([7, 8, 10, 12, 15, 18])
Y = np.array([9, 10, 12, 13, 16, 20])
 
# Training the model
m, b = linear_regression(X, Y)
 
# Making predictions
predictions = predict(X, m, b)
 
# Calculating RMSE
error = rmse(Y, predictions)
 
print("Slope (m):", m)
print("Intercept (b):", b)
print("Predictions:", predictions)
print("RMSE:", error)




Slope (m): 0.9589552238805968
Intercept (b): 2.145522388059705
Predictions: [ 8.85820896  9.81716418 11.73507463 13.65298507 16.52985075 19.40671642]
RMSE: 0.4440037201224643




import matplotlib.pyplot as plt
 
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Independent variable X')
plt.ylabel('Dependent variable y')
plt.title('Linear Regression Model')
plt.legend()
plt.show()