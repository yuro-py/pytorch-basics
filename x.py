import matplotlib.pyplot as plt
import torch
from torch import nn

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

x = torch.arange(start, end, step).unsqueeze(dim=1)
y = (weight * x) + bias

split = int(0.8 * len(x))

x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]


def pred(
    train_data=x_train,
    train_label=y_train,
    test_data=x_test,
    test_label=y_test,
    predictions=None,
):
    plt.figure(figsize=(10, 7), label="Linear Regression Graph")
    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_label, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


pred()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
m0 = LinearRegressionModel()

with torch.inference_mode():
    y_pred = m0(x_test)

pred(predictions=y_pred)
loss_fn = nn.L1Loss()
opti = torch.optim.SGD(params=m0.parameters(), lr=0.01)

epoch_count = []
loss_value = []
test_loss_value = []

torch.manual_seed(42)
epochs = 200

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

        if epoch % 20 == 0:
            epoch_count.append(epoch)
            loss_value.append(loss.item())
            test_loss_value.append(test_loss.item())

            print("------------------------------------")
            print(f"Epoch : {epoch} | Loss : {loss} | Test_Loss : {test_loss}")
            print(m0.state_dict())


def pred_new():
    with torch.inference_mode():
        y_pred_new = m0(x_test)
    pred(predictions=y_pred_new)


pred_new()


def loss_curves():
    plt.figure(figsize=(10, 7), label="Loss curve graph")
    plt.plot(epoch_count, loss_value, c="b", label="Loss value")
    plt.plot(epoch_count, test_loss_value, c="g", label="Test loss value")
    plt.title("Loss count and test loss count")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(prop={"size": 14})
    plt.show()


loss_curves()
