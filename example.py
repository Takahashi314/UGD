import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")
colors = sns.color_palette("Paired", n_colors=12).as_hex()

import numpy as np
import torch

from gmm import GaussianMixture
# from sklearn.mixture import GaussianMixture
from math import sqrt

from utils import run_time


@run_time
def fit(data, d):
    n_components = 3
    model = GaussianMixture(n_components, d)
    model.fit(data)
    # .. used to predict the data points as they where shifted
    y = model.predict(data)


def main():
    # 102400, 512
    n, d = 102400, 512

    # generate some data points ..
    data = torch.Tensor(n, d).normal_().to(torch.float64)
    # .. and shift them around to non-standard Gaussians
    print(data.shape, data.dtype)

    data = data
    fit(data, d)
    print(f"Running on cpu\n")

    data = data.cuda()
    fit(data, d)
    print(f"Running on gpu\n")


def plot(data, y):
    n = y.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(1.61803398875 * 4, 4))
    ax.set_facecolor("#bbbbbb")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # plot the locations of all data points ..
    for i, point in enumerate(data.data):
        if i <= n // 2:
            # .. separating them by ground truth ..
            ax.scatter(*point, color="#000000", s=3, alpha=.75, zorder=n + i)
        else:
            ax.scatter(*point, color="#ffffff", s=3, alpha=.75, zorder=n + i)

        if y[i] == 0:
            # .. as well as their predicted class
            ax.scatter(*point, zorder=i, color="#dbe9ff", alpha=.6, edgecolors=colors[1])
        else:
            ax.scatter(*point, zorder=i, color="#ffdbdb", alpha=.6, edgecolors=colors[5])

    handles = [plt.Line2D([0], [0], color="w", lw=4, label="Ground Truth 1"),
               plt.Line2D([0], [0], color="black", lw=4, label="Ground Truth 2"),
               plt.Line2D([0], [0], color=colors[1], lw=4, label="Predicted 1"),
               plt.Line2D([0], [0], color=colors[5], lw=4, label="Predicted 2")]

    legend = ax.legend(loc="best", handles=handles)

    plt.tight_layout()
    plt.savefig("example.pdf")


if __name__ == "__main__":
    main()
