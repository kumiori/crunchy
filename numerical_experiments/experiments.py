from .core import generate_gaussian_field
from .filters import motion_blur, radial_blur_spin


def run_experiment(shape=(100, 100), blur_distance=30):
    """Run a complete experiment."""
    field = generate_gaussian_field(shape)
    blurred_field = motion_blur(field, blur_distance)
    return blurred_field
