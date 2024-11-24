from dolfinx.mesh import create_rectangle
from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from crunchy.core import generate_gaussian_field
from dolfinx.fem import Function, FunctionSpace
import pyvista as pv

# Create a rectangular DolfinX mesh
nx, ny = image_resolution = (200, 200)  # Same as the image grid resolution
# mesh = create_rectangle(
#     MPI.COMM_WORLD, points=((0.0, 0.0), (1.0, 1.0)), n=(nx, ny), cell_type=2
# )
mesh = create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(nx, ny),
    cell_type=mesh.CellType.triangle,
)
# Tabulate the mesh's degree-of-freedom coordinates
dof_coordinates = mesh.geometry.x
# Create a 2D Gaussian field (as an example image)
x_grid = np.linspace(0, 1, image_resolution[0])
y_grid = np.linspace(0, 1, image_resolution[1])

image = generate_gaussian_field(image_resolution)

from crunchy.filters import radial_blur_zoom

zoom_amount = 30

image = radial_blur_zoom(image, amount=zoom_amount)

# Set up the interpolator
interpolator = RegularGridInterpolator((y_grid, x_grid), image)
# Extract only the x, y coordinates (ignore z)
xy_coordinates = dof_coordinates[:, :2]

# Interpolate the field values at the x, y coordinates
field_values = interpolator(xy_coordinates)

# Create a function space on the mesh (P1 elements)
V = FunctionSpace(mesh, ("CG", 1))  # Continuous Galerkin, degree 1

# Create a Function in this space and assign the sampled values
field_function = Function(V)
field_function.vector.array[:] = field_values
field_function.vector.ghostUpdate()


# Create a VTK representation of the mesh
from dolfinx.plot import vtk_mesh
# ret = compute_topology(mesh, mesh.topology.dim)

# mesh_topology, mesh_cell_types, mesh_coordinates = create_vtk_mesh(mesh)
mesh_topology, mesh_cell_types, mesh_coordinates = vtk_mesh(V)
grid = pv.UnstructuredGrid(mesh_topology, mesh_cell_types, mesh_coordinates)

# Add the scalar field to the grid
grid.point_data["Field"] = field_function.vector.array

# Visualize the field
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="Field", cmap="viridis")
plotter.show()
plotter.screenshot("interpolated_field.png")
