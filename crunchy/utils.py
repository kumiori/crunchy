import matplotlib.pyplot as plt
import os
from pathlib import Path
from mpi4py import MPI


def save_image(image, filename):
    """Save an image to a file."""
    plt.imsave(filename, image, cmap="gray")


def show_image(image):
    """Display an image using matplotlib."""
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def setup_output_directory(storage, parameters, outdir):
    __import__("pdb").set_trace()
    if storage is None:
        prefix = os.path.join(
            outdir, f"1d-{parameters['geometry']['geom_type']}-first-new-hybrid"
        )
    else:
        prefix = storage

    if MPI.COMM_WORLD.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    return prefix
