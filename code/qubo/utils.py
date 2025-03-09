import matplotlib.pyplot as plt
import numpy as np


def plot_solution(solution, dim_size):
    image = np.zeros((dim_size, dim_size), dtype=np.uint8)
    for i in range(dim_size):
        for j in range(dim_size):
            decision = solution[f"x{i * dim_size + j}"]
            image[i, j] = decision  # For visualization purposes.
    print(solution)
    plt.imshow(image)
    plt.show()


def plot_tiff(img, labels, title=""):
    fig, axs = plt.subplots(4, 6, figsize=(16, 16))
    for idx, (ax, label) in enumerate(zip(axs.flatten(), labels)):
        im = np.expand_dims(img[idx], -1).astype(int)
        ax.imshow(im)
        ax.set_title(label)
    axs.flatten()[-1].remove()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_patches(patches, patches_per_dim):
    fig, axes = plt.subplots(patches_per_dim, patches_per_dim, figsize=(10, 10))

    for i in range(patches_per_dim):
        for j in range(patches_per_dim):
            ax = axes[i, j]
            ax.imshow(patches[i, j], cmap="gray")  # Change cmap if needed

    plt.tight_layout()
    plt.show()
