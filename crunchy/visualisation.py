import matplotlib.pyplot as plt


def plot_field(field, title="Field Visualization"):
    """Plot a 2D field."""
    plt.imshow(field, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.show()
