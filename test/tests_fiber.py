from crunchy.mesh import mesh_matrix_fiber_reinforced
from mpi4py import MPI

import dolfinx
from dolfinx.io import XDMFFile, gmshio
from dolfinx.plot import vtk_mesh as compute_topology

comm = MPI.COMM_WORLD
model_rank = 0
# Set geometric parameters and generate the mesh
geom_parameters = {"L": 1.0, "R_inner": 0.2, "lc": 0.05, "geometric_dimension": 2}
gmsh_model, tdim = mesh_matrix_fiber_reinforced(comm, geom_parameters)

# Convert to Dolfinx Mesh
comm = MPI.COMM_WORLD
# mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(
# msh_file, comm=comm, gdim=2
# )
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)
print(mts)
print(fts)
# Visualize using PyVista
import pyvista as pv
import dolfinx.plot

topology, cell_types, geometry = compute_topology(mesh, mesh.topology.dim)

grid = pv.UnstructuredGrid(topology, cell_types, geometry)
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color="lightblue")
plotter.show()
