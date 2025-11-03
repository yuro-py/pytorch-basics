import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

x = torch.arange(start, end, step).unsqueeze(dim=1)
y = bias + (weight * x)

split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

def plot(train_data=x_train, train_label=y_train, test_data=x_test, test_label=y_test, predictions=None):

    plt.figure(figsize=(10,7))

    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_label, c="g", s=4, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

    plt.show()

#plot()

class LinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

# running/testing the class above
torch.manual_seed(42)
model0 = LinearRegressionModel()
print(model0.state_dict())


with torch.inference_mode():
    y_preds = model0(x_test)
print(y_preds)


#plot(predictions=y_preds)

loss_fn = nn.L1Loss
optimizer = torch.optim.SGD(params=model0.parameters(), lr=0.01)


epochs = 1
for epoch in range(epochs):

    model0.train() # set to training mode

    y_pred = model0(x_train) # forward pass

    loss = loss_fn(y_pred, y_train) # calculate the loss

    optimizer.zero_grad() #optimizer zero grad

    loss.backward() # backpropagation on loss, with the parameters

    optimizer.step() # performs gradient descent

    model0.eval() # turns off gradient descent




"""import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

x = torch.arange(start, end, step).unsqueeze(dim=0)
y = bias + (weight * x)

split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

def plot(train_data=x_train, train_label=y_train, test_data=x_test, test_label=y_test, predictions=None):

    plt.figure(figsize=(10,7))

    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_label, c="g", s=4, label="Test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size":14})

    plt.show()

plot()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(1, requires_grade=True, dtype=torch.float32))

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias"""
