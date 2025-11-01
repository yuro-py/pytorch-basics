import torch
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
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = bias + (weight * X)
# print(X[:10], y[:10], len(X), len(y))

# 1. splitting data into training and test-sets(imp in ml)
# university analogy for types/stages of datasets:-
# a. training set = course materials -> model learns patterns from here
# b. validation set = practice exam -> tune model patterns
# c. test set = final exam -> check if model is ready for the wild
# (generalization = ability for a machine learning model to perform well on data it hasn't seen before)

# creating a training split of 80-20:-
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]  # train features & train labels
X_test, y_test = X[split:], y[split:]  # testing features & testing labels

# print(x_train, y_train, x_test, y_test)
# print(len(X_train), len(y_train), len(X_test), len(y_test))


def plot_predictions(
    train_data=X_train,
    train_label=y_train,
    test_data=X_test,
    test_label=y_test,
    predictions=None,
):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

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


# plot_predictions()


# -------------------------------------------------------------------------------------------------------------------------------------------
# 2. Build a model
# pytorch model building essentials:-
# a. torch.nn -> contains all the building blocks of a neural network/computational graph
# b. torch.nn.Parameter -> what parameters should our model try and learn
# c. torch.nn.Module -> the base class for all neural network modules, if you subclass it, you should overwrite forward()
# d. torch.optim -> this is where the optimizers in pytorch live, they will help with gradient descent.
# the optimizers contain the algorithms that optimize the random values in the dataset to represent something meaningful.
# e. def forward() -> all nn.Module subclasses require you to overwrite forward(). this methond defines the computation.
# Create a linear regression model class:-
# WHAT DOES THIS MODEL DO :-
# 1. Start with random values(weight and bias).
# 2. Look at the training data and adjust the random values to better represent
# (or get closer to) the ideal values (the weight and bias) we used to create the data).
#
# Two methods for achieving it(in a combination):-
# 1. gradient descent algorithm
# 2. backpropagation
class LinearRegressionModel(nn.Module):
    # -> nn.Module is the base class of all pytorch classes and features.
    def __init__(self):
        super().__init__()

        # two parameters called weight and bias
        self.weight = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float32)
        )

        # in a small dataset/model we can define weight/bias,
        # but in a real huge dataset/model, this is done by "nn".

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias  # linear regression model

    # any subclass of nn module requires "forward" method for computation.


# gradient descent algorithm -> the randn func will generate random values for weight and biases.
# this algo will adjust the w and b, as close as possible to become a straight line with known parameters.
#
#
# -------------------------------------------------------------------------------------------------------------------------------------------
#
torch.manual_seed(42)

model_0 = LinearRegressionModel()
# print(list(model_0.parameters()))
print(model_0.state_dict())

