import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def point_cloud_show(*args: np.ndarray, labels: List[str] = None) -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    if labels is not None:
        assert len(args) == len(
            labels
        ), "The length between the given `args` and `labels` is different."

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define a colormap to assign unique colors to each iteration
    cmap = plt.get_cmap("plasma")
    num_args = len(args)
    colors = [cmap(i / num_args) for i in range(num_args)]

    for i, point_cloud in enumerate(args):
        x, y, z = point_cloud.T

        label = f"{labels[i]}" if labels is not None else f"point cloud {i + 1}"

        ax.scatter(x, y, z, c=colors[i], marker="o", s=10, label=label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Add a legend to differentiate iterations
    ax.legend()

    plt.show()
