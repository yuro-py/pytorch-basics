import torch
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

    model_0.train()

    print(f"Epoch : {epoch+1}")
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

pred_new()
