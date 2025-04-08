#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from crunchy.core import (
    initialise_functions,
    create_function_spaces_nd,
    setup_output_directory,
    save_parameters,
    initialise_functions,
)

from crunchy.plots import plot_spectrum
from crunchy.mesh import create_extended_rectangle
from dolfinx.io import XDMFFile, gmshio

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
from dolfinx.io import XDMFFile
from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.algorithms.ls import StabilityStepper, LineSearch
from irrevolutions.utils import table_timing_data

from irrevolutions.utils.plots import (
    plot_AMit_load,
    plot_energies,
)
from irrevolutions.utils import (
    Visualization,
    ColorPrint,
    _logger,
)
from dolfinx.fem import (
    Function,
    locate_dofs_topological,
    dirichletbc,
    assemble_scalar,
    form,
)
from ufl import (
    Measure,
)
import matplotlib
from irrevolutions.utils.viz import plot_mesh, plot_profile, plot_scalar, plot_vector
from mpi4py import MPI
from petsc4py import PETSc
from pyvista.plotting.utilities import xvfb
import random
import matplotlib.pyplot as plt
from dolfinx.mesh import CellType, locate_entities, meshtags

print(sys.argv)
petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mesh on node model_rank and then distribute
model_rank = 0


def run_computation(parameters, storage):
    nameExp = parameters["geometry"]["geom_type"]
    geom_params = {
        "L": 1.0,  # Main domain width
        "H": 1.0,  # Main domain height
        "ext": 0.2,  # Extension width around the main domain
        "lc": 0.05,  # Characteristic mesh size
        "tdim": 2,  # Geometric dimension
    }
    with dolfinx.common.Timer(f"~Meshing") as timer:
        gmsh_model, tdim = create_extended_rectangle(comm, geom_params)
        mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)
    __import__("pdb").set_trace()
    dx = Measure("dx", domain=mesh)
    outdir = os.path.join(os.path.dirname(__file__), "output")
    prefix = setup_output_directory(storage, parameters, outdir)

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    save_parameters(parameters, prefix)

    with XDMFFile(
        comm, f"{prefix}/{nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    V_u, V_alpha = create_function_spaces_nd(mesh, dim=2)
    u, u_, alpha, β, v, state = initialise_functions(V_u, V_alpha)

    # Bounds
    alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")
    t = dolfinx.fem.Constant(mesh, np.array(-0.1, dtype=PETSc.ScalarType))
    top_disp = dolfinx.fem.Function(V_u)

    bottom_disp = dolfinx.fem.Constant(
        mesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType)
    )

    for f in [u, alpha_lb, alpha_ub, top_disp]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    # Bcs
    # Locate top and bottom boundaries
    def top_boundary(x):
        return np.isclose(x[1], geom_params["R_outer"], atol=1e-2)

    def bottom_boundary(x):
        return np.isclose(x[1], -geom_params["R_outer"], atol=1e-2)

    boundaries = [(1, top_boundary), (2, bottom_boundary)]

    facet_indices, facet_markers = [], []

    fdim = mesh.topology.dim - 1

    for marker, locator in boundaries:
        facets = locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)

    facet_tag = meshtags(
        mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )

    # debug bcs
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    with XDMFFile(mesh.comm, f"{prefix}/{nameExp}_facet_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tag, mesh.geometry)

    top_dofs = locate_dofs_topological(
        V_u,
        mesh.topology.dim - 1,
        dolfinx.mesh.locate_entities_boundary(mesh, 1, top_boundary),
    )
    bottom_dofs = locate_dofs_topological(
        V_u,
        mesh.topology.dim - 1,
        dolfinx.mesh.locate_entities_boundary(mesh, 1, bottom_boundary),
    )
    bcs_u = [
        # dirichletbc(top_disp, top_dofs, V_u),
        dirichletbc(top_disp, top_dofs),
        dirichletbc(bottom_disp, bottom_dofs, V_u),
    ]
    bcs_alpha = []

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}


def load_parameters(file_path, model="at1"):
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

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yaml"),
        model="at1",
    )
    _storage = f"./output/MPI-{MPI.COMM_WORLD.Get_size()}/{signature[0:6]}"
    visualization = Visualization(_storage)

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data, _, state = run_computation(parameters, _storage)

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
