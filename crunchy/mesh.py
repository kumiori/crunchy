from dolfinx.mesh import create_rectangle
from mpi4py import MPI


def create_2d_mesh(dimensions=(1, 1), resolution=(10, 10)):
    """Generate a simple rectangular mesh."""
    return create_rectangle(
        MPI.COMM_WORLD, points=((0.0, 0.0), dimensions), n=resolution, cell_type=2
    )
