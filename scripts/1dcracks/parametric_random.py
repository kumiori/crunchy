#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
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
from film_stability import run_computation

import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy
from crunchy.utils import save_params_to_yaml, update_parameters, table_timing_data

from mpi4py import MPI

comm = MPI.COMM_WORLD


def load_parameters(file_path, model="at1"):
    """
    Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML parameter file.

    Returns:
        dict: Loaded parameters
        str: Signature of the parameters
    """
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
    parameters["geometry"]["mesh_size_factor"] = 3
    parameters["geometry"]["Lx"] = 3.0
    parameters["geometry"]["Ly"] = 0.1

    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = 0.1
    parameters["model"]["ell_e"] = 0.3
    parameters["model"]["perturbation"] = np.nan

    parameters["loading"]["min"] = 0.99
    parameters["loading"]["max"] = 1.7
    parameters["loading"]["steps"] = 100

    parameters["solvers"]["damage"]["snes"]["snes_monitor"] = None
    parameters["solvers"]["elasticity"]["snes"]["snes_monitor"] = None
    parameters["solvers"]["newton"]["snes_monitor"] = None

    parameters["solvers"]["damage_elasticity"]["max_it"] = 1000
    parameters["solvers"]["damage_elasticity"]["alpha_rtol"] = 1e-4

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    parameter = [0.1, 0.3]

    logging.basicConfig(level=logging.INFO)

    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yaml"),
    )
    prefix = "parametric"

    experimental_data = []
    base_parameters = copy.deepcopy(parameters)
    base_signature = hashlib.md5(str(base_parameters).encode("utf-8")).hexdigest()

    series = base_signature[0::6]
    experiment_dir = os.path.join(prefix, series)
    num_runs = 3

    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    save_params_to_yaml(
        base_parameters, os.path.join(experiment_dir, "parameters.yaml")
    )
    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        for i, t in enumerate(
            np.logspace(np.log10(parameter[0]), np.log10(parameter[1]), num=num_runs)
        ):
            # parameters["model"]["perturbation"] = t
            if changed := update_parameters(parameters, "perturbation", float(t)):
                logging.info(f"Changed perturbation to {t}")
                signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

            print(f"Signature: {signature}")
            history_data = run_computation(parameters, experiment_dir)
            # history_data = np.random.rand(10, 3)
            experimental_data.append(history_data)

    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        experimental_data = pd.DataFrame(history_data)

        if comm.rank == 0:
            experimental_data.to_pickle(f"{experiment_dir}/experimental_data.pkl")
    tasks = [
        # "~Mesh Generation",
        # "~First Order: Equilibrium",
        # "~First Order: AltMin-Damage solver",
        # "~First Order: AltMin-Elastic solver",
        # "~Postprocessing and Vis",
        # "~Output and Storage",
        "~Computation Experiment",
    ]
    timings = table_timing_data(tasks)
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    if comm.rank == 0:
        df = pd.read_pickle(f"{experiment_dir}/experimental_data.pkl")
        print(df)
        print(timings)
        print("Done")
