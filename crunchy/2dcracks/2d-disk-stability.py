#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# from libidris.core import elastic_energy_density_film, damage_energy_density, stress
# from libidris.core import a
# from libidris.core import (
#     setup_output_directory,
#     save_parameters,
#     create_function_spaces_2d,
#     initialize_functions,
# )

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

# from irrevolutions.utils.viz import _plot_bif_spectrum_profiles

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0


def load_parameters(file_path, ndofs, model="at1"):
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
        ndofs=100,
        model="at1",
    )

    # Run computation
    _storage = f"../output/2d-film-second-order-stability-kick/MPI-{MPI.COMM_WORLD.Get_size()}/{signature[0:6]}"
    visualization = Visualization(_storage)

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data, _, state = run_computation(parameters, _storage)

    from irrevolutions.utils import table_timing_data

    tasks = [
        "~First Order: Equilibrium",
        "~First Order: AltMin-Damage solver",
        "~First Order: AltMin-Elastic solver",
        "~Postprocessing and Vis",
        "~Output and Storage",
        "~Computation Experiment",
    ]

    _timings = table_timing_data(tasks)
    visualization.save_table(_timings, "timing_data")
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    ColorPrint.print_bold(f"===================- {signature} -=================")
    ColorPrint.print_bold(f"===================- {_storage} -=================")
