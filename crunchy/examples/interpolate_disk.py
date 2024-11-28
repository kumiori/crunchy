from dolfinx import fem
from mpi4py import MPI
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from crunchy.core import generate_gaussian_field
from dolfinx.fem import Function, FunctionSpace
import pyvista as pv
from irrevolutions.meshes.primitives import mesh_circle_gmshapi
from dolfinx.io import XDMFFile, gmshio
import ufl
from crunchy.utils import show_image

# Create a rectangular DolfinX mesh
nx, ny = image_resolution = (1000, 1000)  # Same as the image grid resolution
# mesh = create_rectangle(
radius = 1.0

comm = MPI.COMM_WORLD

gmsh_model, tdim = mesh_circle_gmshapi(
    "", radius, 0.01, tdim=2, order=1, msh_file=None, comm=MPI.COMM_WORLD
)

mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, 0, tdim)
h = ufl.CellDiameter(mesh)

# Tabulate the mesh's degree-of-freedom coordinates
dof_coordinates = mesh.geometry.x
# Create a 2D Gaussian field (as an example image)
x_grid = np.linspace(-1, 1, image_resolution[0])
y_grid = np.linspace(-1, 1, image_resolution[1])


def generate_linear_gradient(shape):
    """
    Generate a 2D linear gradient in the x[0] direction.

    Parameters:
        shape (tuple): Size of the grid (rows, cols).

    Returns:
        numpy.ndarray: A 2D linear gradient field.
    """
    rows, cols = shape
    gradient = np.linspace(0, 1, cols)  # Linear gradient along x[0] (columns)
    return np.tile(gradient, (rows, 1))  # Repeat for each row


def generate_abs_distance_field(shape):
    """
    Generate a 2D field representing |x - x0|, where x0 is the center of the image.

    Parameters:
        shape (tuple): Shape of the field (rows, cols).

    Returns:
        numpy.ndarray: A 2D field with values |x - x0|.
    """
    rows, cols = shape
    x = np.linspace(0, 1, cols)  # Normalized x-coordinates [0, 1]
    y = np.linspace(0, 1, rows)  # Normalized y-coordinates [0, 1]
    x_grid, y_grid = np.meshgrid(x, y)

    # Center of the image
    x0, y0 = 0.5, 0.5

    # Compute |x - x0| (horizontal distance)
    abs_distance = np.abs(x_grid - x0)

    return abs_distance


image = generate_gaussian_field(image_resolution)
# image = generate_linear_gradient(image_resolution)
# image = generate_abs_distance_field(image_resolution)

from crunchy.filters import (
    radial_blur_zoom,
    radial_blur_spin,
    twirl_effect,
    motion_blur,
    twirl_effect_quadratic,
)

# zoom_amount = 30
blur_distance = spin_amount = 30

# image = motion_blur(image, blur_distance)
# image = twirl_effect_quadratic(image, angle=60)
image = radial_blur_zoom(image, amount=30)

show_image(image)

# import matplotlib.pyplot as plt


# Set up the interpolator
interpolator = RegularGridInterpolator((y_grid, x_grid), image)
# Extract only the x, y coordinates (ignore z)
# xy_coordinates = dof_coordinates[:, :2]
# xy_coordinates = xy_coordinates

# Normalize the circular coordinates to the [-1, 1] range
xy_coordinates = dof_coordinates[:, :2]
#
# Interpolate the field values at the x, y coordinates
field_values = interpolator(xy_coordinates)

# Create a function space on the mesh (P1 elements)
V = fem.functionspace(mesh, ("CG", 1))  # Continuous Galerkin, degree 1

# Create a Function in this space and assign the sampled values
field_function = Function(V)
field_function.x.petsc_vec.array[:] = field_values
field_function.x.petsc_vec.ghostUpdate()

# Create a VTK representation of the mesh
from dolfinx.plot import vtk_mesh
# ret = compute_topology(mesh, mesh.topology.dim)

# mesh_topology, mesh_cell_types, mesh_coordinates = create_vtk_mesh(mesh)
mesh_topology, mesh_cell_types, mesh_coordinates = vtk_mesh(V)
grid = pv.UnstructuredGrid(mesh_topology, mesh_cell_types, mesh_coordinates)

# Add the scalar field to the grid
grid.point_data["Field"] = field_function.x.petsc_vec.array

# Visualize the field
# off screen
plotter = pv.Plotter()
# plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(grid, scalars="Field", cmap="viridis")
# set the title
plotter.add_text("Interpolated Field", font_size=10)
plotter.show()
plotter.screenshot("interpolated_field.png")
