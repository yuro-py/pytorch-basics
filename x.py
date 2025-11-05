import torch
import matplotlib.pyplot as plt
from torch import nn

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


def predplot(
    train_data=x_train,
    train_label=y_train,
    test_data=x_test,
    test_label=y_test,
    predictions=None,
):
    plt.figure(figsize=(6, 6), label="GRAPH")

    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_label, c="g", s=4, label="Test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

    plt.show()


# predplot()


class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
m0 = LinearRegressionModelV2()

with torch.inference_mode():
    y_pred = m0(x_test)
predplot(predictions=y_pred)

loss_fn = nn.L1Loss()
opti = torch.optim.SGD(params=m0.parameters(), lr=0.01)

epoch_count = []
loss_value = []
test_loss_value = []

torch.manual_seed(42)
epochs = 100
for epoch in range(epochs):
    m0.train()

    y_pred = m0(x_train)

    loss = loss_fn(y_pred, y_train)

    opti.zero_grad()

    loss.backward()

    opti.step()

    m0.eval()
    with torch.inference_mode():
        test_pred = m0(x_test)

        test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 1:
            epoch_count.append(epoch)
            loss_value.append(loss.item())
            test_loss_value.append(test_loss.item())

            print("----------")
            print(f"Epoch : {epoch} | Loss : {loss} | Test_Loss : {test_loss}")
            print(m0.state_dict())


def pred_new():
    with torch.inference_mode():
        y_pred_test = m0(x_test)
    predplot(predictions=y_pred_test)


pred_new()


def loss_curves():
    plt.figure(figsize=(7, 7), label="Loss Curve Graph")
    plt.plot(epoch_count, test_loss_value, c="b", label="Test loss")
    plt.plot(epoch_count, loss_value, c="g", label="Train loss")
    plt.title("Training and test loss curve")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


loss_curves()


"""import torch
import matplotlib.pyplot as plt
from torch import nn

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

x = torch.arange(start, end, step).unsqueeze(dim=1)
y = bias + (x * weight)

split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]


def plot(train_data=x_train, train_label=y_train, test_data=x_test, test_label=y_test, predictions=None):

    plt.figure(figsize=(10,7), label="Training graph")

    plt.scatter(train_data, train_label, c="b", s=4, label="Training Data")

    plt.scatter(test_data, test_label, c="g", s=4, label="Testing Data")

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
        return (self.weight * x) + self.bias

torch.manual_seed(42)
model_0 = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model_0(x_test)

plot(predictions=y_preds)

loss_fn = nn.L1Loss()
opti = torch.optim.SGD(params=model_0.parameters(), lr = 0.01)


torch.manual_seed(42)
epochs = int(input("Epochs -> "))

for epoch in range(epochs):

    print(f"Epoch : {epoch+1}")

    print(f"W&B-> {model_0.state_dict()}")

    model_0.train()

    y_preds = model_0(x_train)

    loss = loss_fn(y_preds, y_train)
    print(f"Loss :{loss}")

    opti.zero_grad()

    loss.backward()

    opti.step()

    model_0.eval()
    print("-----------")

def pred_new():
    with torch.inference_mode():
        y_pred_new = model_0(x_test)
        plot(predictions = y_pred_new)

pred_new()"""
