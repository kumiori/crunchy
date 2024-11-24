import matplotlib.pyplot as plt


def save_image(image, filename):
    """Save an image to a file."""
    plt.imsave(filename, image, cmap="gray")


def show_image(image):
    """Display an image using matplotlib."""
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()
