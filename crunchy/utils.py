import matplotlib.pyplot as plt
import os
from pathlib import Path
from mpi4py import MPI
import yaml


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


def save_params_to_yaml(params, filename):
    """
    Save the updated params dictionary to a YAML file.

    Args:
    - params (dict): Dictionary containing all parameters.
    - filename (str): Path to the YAML file to save.
    """
    with open(filename, "w") as file:
        yaml.dump(params, file, default_flow_style=False)


def update_parameters(parameters, key, value):
    """
    Recursively traverses the dictionary d to find and update the key's value.

    Args:
    d (dict): The dictionary to traverse.
    key (str): The key to find and update.
    value: The new value to set for the key.

    Returns:
    bool: True if the key was found and updated, False otherwise.
    """
    if key in parameters:
        parameters[key] = value
        return True

    for k, v in parameters.items():
        if isinstance(v, dict):
            if update_parameters(v, key, value):
                return True

    return False


def table_timing_data(tasks=None):
    import pandas as pd
    from dolfinx.common import timing

    timing_data = []
    if tasks is None:
        tasks = [
            "~Mesh Generation",
            "~First Order: min-max equilibrium",
            "~Postprocessing and Vis",
            "~Computation Experiment",
        ]

    for task in tasks:
        timing_data.append(timing(task))

    df = pd.DataFrame(
        timing_data, columns=["reps", "wall tot", "usr", "sys"], index=tasks
    )

    return df
