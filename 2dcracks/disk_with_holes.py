#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import dolfinx
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import pandas as pd
import petsc4py
import pyvista
import ufl
import yaml
from dolfinx.common import list_timings
from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
from dolfinx.fem.petsc import assemble_vector, set_bc
from dolfinx.io import XDMFFile, gmshio
from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.algorithms.ls import StabilityStepper, LineSearch

from irrevolutions.solvers.function import vec_to_functions
from irrevolutions.utils import (
    ColorPrint,
    ResultsStorage,
    Visualization,
    _logger,
    _write_history_data,
    history_data,
    norm_H1,
    norm_L2,
)
from irrevolutions.utils.plots import (
    plot_AMit_load,
    plot_energies,
    plot_force_displacement,
)
from irrevolutions.utils.viz import plot_mesh, plot_profile, plot_scalar, plot_vector
from irrevolutions.models import BrittleMembraneOverElasticFoundation as ThinFilm
from irrevolutions.meshes.primitives import mesh_circle_gmshapi

from mpi4py import MPI
from petsc4py import PETSc
from pyvista.plotting.utilities import xvfb
from crunchy.core import (
    setup_output_directory,
    save_parameters,
    create_function_spaces_2d,
    initialise_functions,
)

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

from crunchy.mesh import mesh_circle_with_holes_gmshapi


def run_computation(parameters, storage=None):
    _nameExp = parameters["geometry"]["geom_type"]
    R = parameters["geometry"]["R"]
    lc = parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"]

    # Get geometry model
    parameters["geometry"]["geom_type"]

    outdir = os.path.join(os.path.dirname(__file__), "output")
    prefix = setup_output_directory(storage, parameters, outdir)
    signature = save_parameters(parameters, prefix)

    geom_signature = hashlib.md5(
        str(parameters["geometry"]).encode("utf-8")
    ).hexdigest()

    msh_file = f"meshwithholes-{geom_signature}.msh"
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

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    if comm.rank == 0:
        plt.figure()
        ax = plot_mesh(mesh)
        fig = ax.get_figure()
        fig.savefig(f"{prefix}/mesh.png")

    # Functional Setting

    V_u, V_alpha = create_function_spaces_2d(mesh)
    u, u_, alpha, β, v, state = initialise_functions(V_u, V_alpha)

    # Bounds
    alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")

    return history_data


def load_parameters(file_path, model="at1"):
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    return parameters, signature


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)
    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yaml"),
    )
    _storage = f"output/with_holes/MPI-{MPI.COMM_WORLD.Get_size()}/test"
    visualization = Visualization(_storage)

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data = run_computation(parameters, _storage)

    experimental_data = pd.DataFrame(history_data)
    from irrevolutions.utils import table_timing_data

    tasks = [
        "~First Order: Equilibrium",
        "~First Order: AltMin-Damage solver",
        "~First Order: AltMin-Elastic solver",
        "~Postprocessing and Vis",
        "~Output and Storage",
        "~Computation Experiment",
    ]
    _timings = table_timing_data()

    visualization.save_table(_timings, "timing_data")
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    ColorPrint.print_bold(f"===================- {signature} -=================")
    ColorPrint.print_bold(f"===================- {_storage} -=================")
