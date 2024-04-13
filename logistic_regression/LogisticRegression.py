import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import datasets
import warnings

warnings.filterwarnings("ignore")

class LogisticRegression():
    def __init__(self, lr=0.001, n_iters = 1000):
        self.lr = lr 
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights)
            y_pred = self.sigmoid(y_pred)
            dw = 1 / n_samples * np.dot(X.T, (y_pred - y))
            db = 2 * (y_pred - y)
            
            # Update weights
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db


    def predict(self, X):
        y_pred = np.dot(X, self.weights)
        y_pred = self.sigmoid(y_pred)
        return [0 if y < 0.5 else 1 for y in y_pred]


    def sigmoid(self, X):
        sig = 1 / (1 + np.exp(-X))
        return sig


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/ len(y_test)

if __name__ == "__main__":
    model = LogisticRegression()
    bc = datasets.load_breast_cancer()
    features = bc.feature_names
    targets = bc.target_names
    print("Creating logistic regression model on brest cancer data with\n"
            + f"{len(features)} features and {len(targets)} target values")
    
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    print("Fitting...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy of the model is: {100 * accuracy(y_pred, y_test):.2f}%")
    
    