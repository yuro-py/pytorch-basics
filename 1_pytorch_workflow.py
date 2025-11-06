# DON'T TOUCH BELOW TWO LINES OF CODE
import time

start_time = time.perf_counter()
# DON'T TOUCH ABOVE TWO LINES OF CODE

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

# device = "cuda" if torch.cuda.is_available() else "cpu"

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


def plot_predictions(
    train_data=x_train,
    train_label=y_train,
    test_data=x_test,
    test_label=y_test,
    predictions=None,
):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_label, c="g", s=4, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()


plot_predictions()

# WHAT DOES THIS MODEL DO :-
# 1. Start with random values(weight and bias).
# 2. Look at the training data and adjust the random values to better represent
# (or get closer to) the ideal values (the weight and bias) we used to create the data).


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
model_0 = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model_0(x_test)

plot_predictions(predictions=y_preds)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

torch.manual_seed(42)
epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_0.train()

    # 1. forward pass
    y_pred = model_0(x_train)

    # 2.calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. optimizer zero grad
    optimizer.zero_grad()

    # 4. backpropagation on the loss
    loss.backward()

    # 5. step the optimizer(perform gradient descent)
    optimizer.step()

    # turns off gradient tracking
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(x_test)

        test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 1:
            epoch_count.append(epoch)
            loss_values.append(loss.item())
            test_loss_values.append(test_loss.item())

            # print("----------------------------------")
            # print(f"Epoch :{epoch} | Loss : {loss} | Test_loss : {test_loss}")
            # print(model_0.state_dict())


def pred_new():
    with torch.inference_mode():
        y_pred_test = model_0(x_test)
    plot_predictions(predictions=y_pred_test)


pred_new()


def loss_curves():
    plt.figure(figsize=(6, 6), label="Loss Curve Graph")
    plt.plot(epoch_count, test_loss_values, c="b", label="Test loss")
    plt.plot(epoch_count, loss_values, c="g", label="Train Loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


loss_curves()


# DON'T TOUCH : MODEL SAVING/LOADING SYSTEM
# 1. create models dir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# 2. create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# print(f"Saving model to : {MODEL_SAVE_PATH}")
# torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

# loaded_model_0 = LinearRegressionModel()
# loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
# DON'T TOUCH : MODEL SAVING/LOADING SYSTEM

# Running and seeing the saved model
# loaded_model_0.eval()
# with torch.inference_mode():
#    loaded_model_preds = loaded_model_0(x_test)
# print(loaded_model_preds)


# model_0.eval()
# with torch.inference_mode():
#    y_preds = model_0(x_test)
# print(y_preds)

# print(loaded_model_preds == y_preds)

# DON'T TOUCH : MODEL RUNNING TIME CALCULATOR
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
# DON'T TOUCH ABOVE THREE LINES OF CODE
