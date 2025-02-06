import hashlib
import logging
import os
import numpy as np
import yaml
import dolfinx
from dolfinx.fem import (
    Function,
    assemble_scalar,
    form,
    locate_dofs_geometrical,
    dirichletbc,
)
from ufl import Measure as measure
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.utils import setup_logger_mpi, _logger, Visualization, ColorPrint
from dolfinx.common import list_timings

from crunchy.core import (
    initialise_functions,
    create_function_spaces_nd,
    setup_output_directory,
    save_parameters,
    initialise_functions,
)
from irrevolutions.models.one_dimensional import FilmModel1D as ThinFilm

comm = MPI.COMM_WORLD

# extend class ThinFilm to include AT2 damage model


def run_computation(parameters, storage=None, logger=_logger):
    Lx = parameters["geometry"]["Lx"]
    _nameExp = parameters["geometry"]["geom_type"]
    parameters["model"]["ell"]

    # Get geometry model
    parameters["geometry"]["geom_type"]
    N = parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"]

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, int(1 / N))
    outdir = os.path.join(os.path.dirname(__file__), "output")

    # prefix = setup_output_directory(storage, parameters, outdir)
    prefix = storage

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    save_parameters(parameters, prefix)

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    # Functional Setting
    V_u, V_alpha = create_function_spaces_nd(mesh, dim=1)
    u, u_, alpha, β, v, state = initialise_functions(V_u, V_alpha)

    # Bounds
    alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")

    u_zero = Function(V_u, name="InelasticDisplacement")
    eps_t = dolfinx.fem.Constant(mesh, np.array(1.0, dtype=PETSc.ScalarType))
    u_zero.interpolate(lambda x: eps_t / 2.0 * (2 * x[0] - Lx))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for f in [u, u_zero, alpha_lb, alpha_ub]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
    # Redundant boundary conditions
    bcs_u = [dirichletbc(u_zero, dofs_u_right), dirichletbc(u_zero, dofs_u_left)]
    dx = measure("dx", domain=mesh)

    bcs_alpha = []
    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
    model = ThinFilm(parameters["model"])
    total_energy = (
        model.elastic_energy_density(state, u_zero) + model.damage_energy_density(state)
    ) * dx

    loads = np.linspace(
        parameters["loading"]["min"],
        parameters["loading"]["max"],
        parameters["loading"]["steps"],
    )

    hybrid = HybridSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    bifurcation = BifurcationSolver(
        total_energy, state, bcs, bifurcation_parameters=parameters.get("stability")
    )

    stability = StabilitySolver(
        total_energy, state, bcs, cone_parameters=parameters.get("stability")
    )

    logging.basicConfig(level=logging.INFO)


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

    if model == "at2":
        parameters["loading"]["min"] = 0.0
        parameters["loading"]["max"] = 3.0
        parameters["loading"]["steps"] = 30

    parameters["geometry"]["geom_type"] = "1d-bar"
    parameters["geometry"]["mesh_size_factor"] = 4
    parameters["geometry"]["N"] = ndofs

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
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
    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "umut_benchmark_parameters.yaml"),
        ndofs=100,
        model="at2",
    )
    # Run computation
    _storage = f"output/umut_benchmark_1d/{signature[0:6]}"

    visualization = Visualization(_storage)
    prefix = setup_output_directory(_storage, parameters, "")
    logger = setup_logger_mpi(filename=f"{_storage}/evolution.log")

    with dolfinx.common.Timer("~Computation Experiment") as timer:
        history_data, stability_data, state = run_computation(parameters, _storage)

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
    _timings = table_timing_data()
    visualization.save_table(_timings, "timing_data")
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    ColorPrint.print_bold(f"===================- {signature} -=================")
    ColorPrint.print_bold(f"===================- {_storage} -=================")
