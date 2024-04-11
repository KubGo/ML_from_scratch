import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from PIL import Image
import os


def mse(y_test, y_pred):
    return np.mean((y_test - y_pred) ** 2)

class LinearRegression:
    def __init__(self, learning_rate = 0.001, n_iterations=1000) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
        
            y_pred = self.predict(X)
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
        
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    def calculate_mse(self, X, y):
        y_pred = self.predict(X)
        metric = mse(y, y_pred)
        return metric
    
# Test to show how the linear regression works
if __name__ == "__main__":
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1234)
    lr = LinearRegression(learning_rate=0.01, n_iterations=2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    lr.fit(X_train, y_train)
    mse_text = f"MSE in training data: {lr.calculate_mse(X_train, y_train):.2f}\nMSE in testing data: {lr.calculate_mse(X_test, y_test):.2f}"
    cmap = plt.get_cmap('viridis')
    plt.text(-3.5, 200, mse_text, bbox={
        'alpha': 0.5,
        "pad": 5})
    training_points = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    testing_points = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, lr.predict(X), color='red', linewidth=2, label="Prediction")
    plt.savefig("Linear_Regression.jpg")
    img = Image.open("Linear_Regression.jpg")
    img.show()
    os.remove("Linear_Regression.jpg")