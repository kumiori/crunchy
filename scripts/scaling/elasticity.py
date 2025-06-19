#!/usr/bin/env python3
import json
import logging
import os
import sys
from pathlib import Path

import dolfinx
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import petsc4py
import pyvista
import ufl
import yaml
from dolfinx import log
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
from pyvista.plotting.utilities import xvfb
from irrevolutions.meshes.primitives import mesh_bar_gmshapi
from irrevolutions.models import ElasticityModel
from irrevolutions.solvers import SNESSolver as ElasticitySolver
from irrevolutions.utils.viz import plot_vector
import basix.ufl
import argparse
from dolfinx.common import list_timings

logging.basicConfig(level=logging.INFO)

import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy
import math

# ///////////

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0


# with open(os.path.join(os.path.dirname(__file__), "parameters.yml")) as f:
# with pkg_resources.path("irrevolutions.models", "default_parameters.yml") as path:
# with open(path, "r") as f:


def get_number_of_dofs(mesh, element_type="CG", degree=1):
    """
    Get the total number of degrees of freedom (DOFs) in a dolfinx mesh,
    accounting for parallel execution and block size.

    Parameters:
    - mesh (dolfinx.mesh.Mesh): The input mesh object.
    - element_type (str): The type of finite element ("CG" for continuous Galerkin by default).
    - degree (int): The polynomial degree of the element (default is 1).

    Returns:
    - total_dofs (int): Total number of degrees of freedom across all processes.
    """

    # Define the function space
    V = dolfinx.fem.functionspace(mesh, (element_type, degree))

    # Get the block size
    # Local DOFs on the current process
    local_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    # Sum across all processes to get the global number of DOFs
    total_dofs = MPI.COMM_WORLD.allreduce(local_dofs, op=MPI.SUM)

    return total_dofs


def run_elasticity(parameters):
    # Get mesh parameters
    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    tdim = parameters["geometry"]["geometric_dimension"]
    lc = parameters["geometry"]["lc"]

    # Get geometry model
    geom_type = parameters["geometry"]["geom_type"]

    # Create the mesh of the specimen with given dimensions
    with dolfinx.common.Timer(f"~Meshing") as timer:
        gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

        # Get mesh and meshtags
        mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    total_dofs = get_number_of_dofs(mesh)

    outdir = os.path.join(os.path.dirname(__file__), "output")
    prefix = os.path.join(outdir, "elasticity")

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    with XDMFFile(comm, f"{prefix}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)

    # Function spaces
    element_u = basix.ufl.element(
        "Lagrange", mesh.basix_cell(), degree=1, shape=(tdim,)
    )
    V_u = dolfinx.fem.functionspace(mesh, element_u)
    V_ux = dolfinx.fem.functionspace(
        mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
    )

    # Define the state
    u = dolfinx.fem.Function(V_u, name="Displacement")
    u_ = dolfinx.fem.Function(V_u, name="Boundary Displacement")
    ux_ = dolfinx.fem.Function(V_ux, name="Boundary Displacement")
    zero_u = dolfinx.fem.Function(V_u, name="   Boundary Displacement")

    state = {"u": u}

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    dofs_u_left = dolfinx.fem.locate_dofs_geometrical(
        V_u, lambda x: np.isclose(x[0], 0.0)
    )
    dofs_u_right = dolfinx.fem.locate_dofs_geometrical(
        V_u, lambda x: np.isclose(x[0], Lx)
    )
    dofs_ux_right = dolfinx.fem.locate_dofs_geometrical(
        V_ux, lambda x: np.isclose(x[0], Lx)
    )

    # Set Bcs Function
    zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
    u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
    ux_.interpolate(lambda x: np.ones_like(x[0]))

    for f in [zero_u, ux_]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    bcs_u = [
        dolfinx.fem.dirichletbc(zero_u, dofs_u_left),
        dolfinx.fem.dirichletbc(u_, dofs_u_right),
        # dolfinx.fem.dirichletbc(ux_, dofs_ux_right, V_u.sub(0)),
    ]

    bcs = {"bcs_u": bcs_u}
    # Define the model
    model = ElasticityModel(parameters["model"])

    # Energy functional
    f = dolfinx.fem.Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work
    energy_u = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

    solver = ElasticitySolver(
        energy_u,
        u,
        bcs_u,
        bounds=None,
        petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
        prefix=parameters.get("solvers").get("elasticity").get("prefix"),
    )

    history_data = {
        "load": [],
        "elastic_energy": [],
    }

    for i_t, t in enumerate(loads):
        u_.interpolate(lambda x: (t * np.ones_like(x[0]), 0 * np.ones_like(x[1])))
        u_.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        logging.info(f"-- Solving for t = {t:3.2f} --")

        with dolfinx.common.Timer(f"~Elasticity solver") as timer:
            solver.solve()

        elastic_energy = comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(model.elastic_energy_density(state) * dx)
            ),
            op=MPI.SUM,
        )

        history_data["load"].append(t)
        history_data["elastic_energy"].append(elastic_energy)

        with XDMFFile(
            comm, f"{prefix}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        ) as file:
            file.write_function(u, t)

        if comm.rank == 0:
            a_file = open(f"{prefix}_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()


def compute_mesh_size(num_dofs, L, H):
    """
    Compute the mesh size for a 2D rectangular specimen based on the number of degrees of freedom (DOFs).

    Parameters:
    - num_dofs (int): Total number of degrees of freedom.
    - L (float): Length of the rectangular specimen.
    - H (float): Height of the rectangular specimen.

    Returns:
    - lc (float): Mesh size.
    """
    # Preserve aspect ratio of the rectangle
    aspect_ratio = H / L
    lc = math.sqrt(2 * aspect_ratio * L / num_dofs)

    return lc


def load_parameters(file_path, args):
    """
    Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML parameter file.

    Returns:
        dict: Loaded parameters.
    """
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["loading"]["min"] = 1.0
    parameters["loading"]["max"] = 1.0
    parameters["loading"]["steps"] = 1

    default_Lx = 1.0
    default_Ly = 0.1

    # Check the geometry type
    if parameters.get("geometry_type") == "bar":
        # Attempt to get dimensions from parameters
        Lx = parameters.get("geometry").get("Lx", default_Lx)
        Ly = parameters.get("geometry").get("Ly", default_Ly)
    else:
        # Use default values for other geometry types
        Lx = parameters["geometry"]["Lx"] = default_Lx
        Ly = parameters["geometry"]["Ly"] = default_Ly

    # Compute mesh size based on the number of degrees of freedom
    lc = compute_mesh_size(args.problem_size, Lx, Ly)

    parameters["geometry"]["lc"] = lc

    parameters["solvers"]["elasticity"]["snes"]["snes_monitor"] = None
    parameters["solvers"]["elasticity"]["snes"]["snes_maxit"] = args.max_iterations
    parameters["solvers"]["elasticity"]["snes"]["snes_atol"] = args.tolerance

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple elasticity problem.")

    # General problem parameters
    parser.add_argument(
        "--problem-size",
        type=int,
        required=True,
        help="Size of the problem (e.g., number of dofs).",
        default=1000,
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        required=True,
        help="Number of processors to use.",
        default=1,
    )

    # Solver options
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, help="Solver absolute tolerance."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of solver iterations.",
    )

    args = parser.parse_args()

    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yaml"), args
    )
    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data = run_elasticity(parameters)

    from irrevolutions.utils import table_timing_data

    tasks = [
        "~Elasticity solver",
        "~Computation Experiment",
    ]

    _timings = table_timing_data(tasks)
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
