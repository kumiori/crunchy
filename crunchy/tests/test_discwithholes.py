from dolfinx import mesh, fem
from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile
import numpy as np
from crunchy.mesh import (
    mesh_circle_with_holes_gmshapi,
)
from petsc4py import PETSc
import matplotlib.pyplot as plt


def import_mesh_with_hole_markers(gmsh_model, comm=MPI.COMM_WORLD):
    # Export the Gmsh model to a file
    msh_file = "circle_with_holes.msh"
    # gmsh_model.write(msh_file)

    # Read the mesh using meshio
    msh = gmshio.read_from_msh(msh_file, comm=MPI.COMM_WORLD)

    # Create the Dolfinx mesh
    dolfinx_mesh = mesh.create_mesh(
        MPI.COMM_WORLD, msh.cells_dict["triangle"], msh.points
    )

    # Extract boundary markers
    facet_data = msh.cell_data_dict["line"]["gmsh:physical"]

    return dolfinx_mesh, facet_data


# Example Usage
comm = MPI.COMM_WORLD
R, lc, tdim = 1.0, 0.1, 2
num_holes = 0
hole_radius = 0.1

# Generate the Gmsh model
# gmsh_model, tdim = mesh_circle_with_holes_gmshapi(
#     "CircleWithHoles",
#     R,
#     lc,
#     tdim,
#     num_holes,
#     hole_radius,
# )
# model_rank = 0

from irrevolutions.utils.viz import plot_mesh
import os

# gmsh_model, tdim = mesh_circle_gmshapi(
#     "disc", R, lc, tdim=2, order=1, msh_file=None, comm=MPI.COMM_WORLD
# )

# if meshwithholes.msh exists load it
# otherwise create it

msh_file = "meshwithholes.msh"
if not os.path.exists(msh_file):
    gmsh_model, tdim = mesh_circle_with_holes_gmshapi(
        "discwithholes",
        R,
        lc,
        tdim=2,
        num_holes=3,
        hole_radius=0.05,
        hole_positions=None,
        refinement_factor=0.8,
        order=1,
        msh_file=msh_file,
        comm=MPI.COMM_WORLD,
    )

mesh, cell_tags, facet_tags = gmshio.read_from_msh(msh_file, comm=MPI.COMM_WORLD)
# mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, 0, tdim)
mesh.topology.create_connectivity(tdim - 1, tdim)

if comm.rank == 0:
    plt.figure()
    ax = plot_mesh(mesh)
    fig = ax.get_figure()
    fig.savefig(f"meshwithholes.png")

# Import mesh and facet markers
# dolfinx_mesh, facet_data = import_mesh_with_hole_markers(gmsh_model)

# Apply zero Dirichlet condition on the hole boundaries
import basix
from dolfinx.fem import functionspace

element = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))
V = functionspace(mesh, element)

bc_values = fem.Constant(mesh, PETSc.ScalarType([0.0, 0.0]))
bcs = []

boundary_dofs = fem.locate_dofs_topological(
    V, mesh.topology.dim - 1, facet_tags.indices
)
bc = fem.dirichletbc(bc_values, boundary_dofs, V)

__import__("pdb").set_trace()
