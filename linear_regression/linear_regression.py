import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Import train and test data
test_data = pd.read_csv('data/test.csv')
train_data = pd.read_csv('data/train.csv')

# Rid training and test data of dirty(NaN) values
train_data = train_data.dropna()
test_data = test_data.dropna()

# X and Y columns from test data
x_test = test_data.as_matrix(['x'])
y_test = test_data.as_matrix(['y'])

# X and Y columns from training data
x = train_data.as_matrix(['x'])
y = train_data.as_matrix(['y'])

# Fitting model to X and Y of training data
model = LinearRegression()
model.fit(x, y)

# Prediction of Y from X of test data
predictions = model.predict(x_test)

# Visualizing training set results
plt.scatter(x, y, color='red')
plt.plot(x, model.predict(x), color='blue')
plt.title('Y vs X (Training set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Visualizing test set results
plt.scatter(x_test, y_test, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title('Y vs X (Test set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

pred_y_mean = np.mean(predictions)
actual_y_mean = np.mean(y_test)

print("[+] Predicted Y mean: ", pred_y_mean)
print("[+] Actual Y mean: ", actual_y_mean)
print("[-] Absolute mean error: ", actual_y_mean - pred_y_mean)
