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
from umut_validation_at2 import run_computation

import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy
from crunchy.utils import save_params_to_yaml, update_parameters, table_timing_data

from mpi4py import MPI

comm = MPI.COMM_WORLD


def load_parameters(file_path, ndofs, model="at2"):
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

    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = "1D"
    parameters["geometry"]["geom_type"] = "thinfilm"
    # Get mesh parameters

    # if model == "at2":
    parameters["loading"]["min"] = 0.0
    parameters["loading"]["max"] = 3.5
    parameters["loading"]["steps"] = 50

    parameters["geometry"]["geom_type"] = "1d-bar"
    parameters["geometry"]["mesh_size_factor"] = 4
    parameters["geometry"]["N"] = ndofs

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-5
    parameters["stability"]["cone"]["cone_rtol"] = 1e-5
    parameters["stability"]["cone"]["scaling"] = 1e-3

    parameters["model"]["w1"] = 1
    parameters["model"]["at_number"] = 2
    parameters["model"]["ell"] = 0.158114
    parameters["model"]["k_res"] = 0.0
    parameters["model"]["mu"] = 1
    parameters["model"]["kappa"] = (0.34) ** (-2)

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    # ell_e 0.1, 0.3
    # kappa = 1./ell**2

    parameter = [0.1, 0.3]

    logging.basicConfig(level=logging.INFO)

    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parametric_stiffness.yaml"),
        ndofs=101,
        model="at2",
    )
    prefix = "output/parametric_stiffness"

    experimental_data = []
    base_parameters = copy.deepcopy(parameters)
    base_signature = hashlib.md5(str(base_parameters).encode("utf-8")).hexdigest()

    series = base_signature[0::6]
    experiment_dir = os.path.join(prefix, series)
    num_runs = 5

    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    save_params_to_yaml(
        base_parameters, os.path.join(experiment_dir, "parameters.yaml")
    )
    parameter_sweep = [
        1 / ell**2 for ell in np.linspace(parameter[0], parameter[1], num=num_runs)
    ]

    for i, t in enumerate(parameter_sweep):
        # parameters["model"]["perturbation"] = t
        if changed := update_parameters(parameters, "kappa", float(t)):
            logging.info(f"Changed kappa to {t}")
            signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
        else:
            logging.info(f"Failed to change parameter to {t}")
            __import__("pdb").set_trace()

        print(f"Signature: {signature}")
        with dolfinx.common.Timer(f"~Computation Experiment") as timer:
            if comm.rank == 0:
                Path(os.path.join(experiment_dir, signature[0:6])).mkdir(
                    parents=True, exist_ok=True
                )

            history_data, stability_data, state = run_computation(
                parameters, os.path.join(experiment_dir, signature[0:6])
            )
            # history_data = np.random.rand(10, 3)

        # experimental_data.append(history_data)

        with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
            experimental_data = pd.DataFrame(history_data)

            if comm.rank == 0:
                experimental_data.to_pickle(f"{experiment_dir}/experimental_data.pkl")

    tasks = [
        "~First Order: Equilibrium",
        "~First Order: AltMin-Damage solver",
        "~First Order: AltMin-Elastic solver",
        "~Postprocessing and Vis",
        "~Output and Storage",
        "~Computation Experiment",
    ]
    timings = table_timing_data(tasks)
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    if comm.rank == 0:
        df = pd.read_pickle(f"{experiment_dir}/experimental_data.pkl")
        print(df)
        print(timings)
        print("Done")
