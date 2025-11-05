# DON'T TOUCH BELOW TWO LINES OF CODE
import time

start_time = time.perf_counter()
# DON'T TOUCH ABOVE TWO LINES OF CODE

from pathlib import Path
import torch
import numpy
from torch import nn  # nn contains all of pytorch building blocks for Neural Networks
import matplotlib.pyplot as plt


# device = "cuda" if torch.cuda.is_available() else "cpu"

# one of many pytorch's workflow:-
# 1. Get data ready(turn to tensors)
# 2. build/pick a pre-trained model
#     2.1 pick a loss function and optimizer
#     2.2 build a training loop
# 3. fit the model to a data and make a prediction, or TRAINING
# 4. evaluate the model
# 5. improve through experimentation
# 6. save and reload your trained model

# -------------------------------------------------------------------------------------------------------------------------------------------
# 1. PREPARING/LOADING DATA
# LINEAR REGRESSION FORMULA :-
# Y = a + bX
# use the formula to make a straight line with known parameters(parameter is something that a model learns)

# creating known parameters:-
weight = 0.7  # as 'a' in the formula
bias = 0.3  # as 'b' in the formula

# create sample dataset
start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = bias + (weight * x)
# print(X[:10], y[:10], len(X), len(y))

# 1. splitting data into training and test-sets(imp in ml)
# university analogy for types/stages of datasets:-
# a. training set = course materials -> model learns patterns from here
# b. validation set = practice exam -> tune model patterns
# c. test set = final exam -> check if model is ready for the wild
# (generalization = ability for a machine learning model to perform well on data it hasn't seen before)

# creating a training split of 80-20:-
split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]  # train features & train labels
x_test, y_test = x[split:], y[split:]  # testing features & testing labels

# print(x_train, y_train, x_test, y_test)
# print(len(X_train), len(y_train), len(X_test), len(y_test))


def plot_predictions(
    train_data=x_train,
    train_label=y_train,
    test_data=x_test,
    test_label=y_test,
    predictions=None,
):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(5, 5))

    # Plot training data in blue
    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_label, c="g", s=4, label="Test data")

    # are there predictions?
    if predictions is not None:
        # plot if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # show the legend
    plt.legend(prop={"size": 14})

    plt.show()


plot_predictions()


# -------------------------------------------------------------------------------------------------------------------------------------------
# 2. Build a model
# pytorch model building essentials:-
# a. torch.nn -> contains all the building blocks of a neural network/computational graph
# b. torch.nn.Parameter -> what parameters should our model learn
# c. torch.nn.Module -> the base class for all neural network modules, if you use it as a subclass, you should overwrite forward()
# d. torch.optim -> this is where the optimizers in pytorch live, they will help with gradient descent.
# the optimizers contain the algorithms that optimize the random values in the dataset to represent something meaningful.
# e. def forward() -> all nn.Module subclasses require you to overwrite forward(). this methond defines the computation.
# Create a linear regression model class:-
# WHAT DOES THIS MODEL DO :-
# 1. Start with random values(weight and bias).
# 2. Look at the training data and adjust the random values to better represent
# (or get closer to) the ideal values (the weight and bias) we used to create the data).


# Two methods for achieving it(in a combination):-
# 1. gradient descent algorithm
# 2. backpropagation


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


# gradient descent algorithm -> the randn func will generate random values for weight and biases.
# this algo will adjust the w and b, as close as possible to become a straight line with known parameters.


torch.manual_seed(42)
model_0 = LinearRegressionModel()
# print(model_0.state_dict())  # lists the model's parameters

# Making predictions using torch.inference_mode(). using it turns off requires_grad.
# testing prediction for "y_test" based on "x_test"
with torch.inference_mode():
    y_preds = model_0(x_test)
# print(y_preds)

plot_predictions(predictions=y_preds)

# -------------------------------------------------------------------------------------------------------------------------------------------
# 3. Train a model(building intuition):-
# a way to measure how poor or how wrong a model's predictions are,
# one way is to use a "loss function".
# Things we need to train:-
# 1. lossfunction : it checks how wrong our output is compared to the ideal output. lower is better.
# 2. optimizer : takes the loss and adjusts the model's parameters.

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
# a. SGD = one of the many algos for optimizing.
# b. lr = Learning rate. It decides how much value to modify in
# the parameters during optimizing. deciding perfect value comes with time.

# inside an optimizer, we need to tell the parameters we would like to modidy using "params",
# and the learning rate has to be specified as well.

# loss value = MAE (mean absolute error)
# -------------------------------------------------------------------------------------------------------------------------------------------
# 4. Building a training/testing loop.
# a. loop through the data.
# b. forward(or forward propagation) pass to make predictions-> this involves data moving through the model's forward() functions
# c. calculate the loss(compare forward pass to "ground truth labels")
# d. optimizer zero grad
# e. loss backwards -> move backwards thru nn to calc the grads of each params with respect to the loss.
# f. optimizer step -> use the optim to adjust the params to improve the loss(gradient descent)

torch.manual_seed(42)

epochs = 200
# an epoch is one loop through the data

# Tracking experience
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    # set the model to training mode. this mode sets all params that require gradients to "require gradients".
    model_0.train()

    # 1. forward pass
    y_pred = model_0(x_train)

    # 2.calculate the loss
    loss = loss_fn(y_pred, y_train)
    # print(f"Loss : {loss}")

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


# re-run and prediction and test the graph changes:-
def pred_new():
    with torch.inference_mode():
        y_pred_test = model_0(x_test)
    plot_predictions(predictions=y_pred_test)


pred_new()

# print(epoch_count)
# print(loss_values)
# print(test_loss_values)


# Plotting loss curves
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
