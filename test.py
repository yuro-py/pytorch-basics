import torch
import matplotlib.pyplot as plt

# linear regression formula :-
# y = a + bX
# set the weight(a) and bias(b):
weight = 0.7
bias = 0.3

# create dataset
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = bias + (weight * X)

# create a training split
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

def plot(train_data=X_train, train_label=y_train, test_data=X_test, test_label=y_test, predictions=None):

    plt.figure(figsize=(10,7))

    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_label, c="g", s=4, label="Test data")
    if predictions is not None:    
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()

plot()
