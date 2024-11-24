import numpy as np


def generate_gaussian_field(shape=(100, 100), mean=0, std=1):
    """Generate a 2D Gaussian field."""
    return np.random.normal(mean, std, size=shape)
