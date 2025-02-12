import gzip
import os

import matplotlib.pyplot as plt
import numpy as np


def load_mnist(path, kind="train"):
    """
    Load MNIST data from `path`
    """

    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


def display_image_array(trainloader):
    """
    Displays the fashion mnist images in a grid with labels
    """

    images, labels = next(iter(trainloader))
    fig = plt.figure(figsize=(15, 5))
    for id in np.arange(20):
        ax = fig.add_subplot(4, 20 // 4, id + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[id]), cmap="gray")
        ax.set_title(labels[id].item())
        fig.tight_layout()
