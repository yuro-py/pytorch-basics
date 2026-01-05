import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.datasets import make_circles

# neural network classification
# classification is the problem of predicting whether soemthing is one thing or another(there can be multiple things as options)

# make classification data and get it ready

n_samples = 1000

x, y = make_circles(n_samples, noise=0.03, random_state=42)

circles = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "label": y})
print(circles.head(10))

plt.figure(figsize=(10, 7))
plt.scatter(x=x[:, 0], y=x[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

# video 9hr19m
