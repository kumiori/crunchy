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
    # elastic_energy_density_film,
    # damage_energy_density,
    # stress,
    # a,
    setup_output_directory,
    save_parameters,
    create_function_spaces_2d,
    initialise_functions,
)

# from irrevolutions.utils.viz import _plot_bif_spectrum_profiles
from crunchy.core import generate_gaussian_field
from scipy.interpolate import RegularGridInterpolator

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

BINARY_DATA = True


def plot_spectrum(history_data):
    """
    Plot the spectrum (eigenvalues) of eigs_ball and eigs_cone for each load step.

    Parameters:
        history_data (dict): Dictionary containing load steps and eigenvalue data.
    """
    # Extract load steps and eigenvalue data
    load_steps = history_data.get("load", [])
    eigs_ball = history_data.get("eigs_ball", [])
    eigs_cone = history_data.get("eigs_cone", [])

    # Ensure we have data for plotting
    if not load_steps or not eigs_ball:
        print("No data to plot.")
        return

    # Iterate over load steps
    fig, ax = plt.subplots(figsize=(10, 6))

    for step_idx, load_step in enumerate(load_steps):
        if eigs_ball:  # Ensure it's not empty
            _eigs_ball_step = eigs_ball[step_idx]
            ax.scatter(
                [load_step] * len(_eigs_ball_step),
                _eigs_ball_step,
                # label=f"eigs_ball (step {step_idx})",
                color="blue",
                alpha=0.7,
            )
        ax.axhline(y=0, color="black", linestyle="--")

        if not np.all(np.isnan(eigs_cone)):  # Ensure it's not empty
            _eigs_cone_step = eigs_cone[step_idx]
            if type(_eigs_cone_step) is float:
                _eigs_cone_step = [_eigs_cone_step]
            ax.scatter(
                [load_step] * len(_eigs_cone_step),
                _eigs_cone_step,
                marker="x",
                # label=f"eigs_ball (step {step_idx})",
                color="red",
                alpha=0.7,
            )
        # Add plot labels and legend
        # ax.title(f"Spectrum at Load Step {step_idx} (Load: {load_step})")
        # ax.set_xlabel("Eigenvalue Index")
        # ax.set_ylabel("Eigenvalue Magnitude")
        # ax.legend()
        # plt.grid(True)

        # Save the figure for each step
        # fig.savefig(f"spectrum.png")
        # plt.close(fig)
        # print(
        # f"Saved spectrum plot for load step {step_idx} as spectrum_step_{step_idx}.png"
        # )
        # fig.close()

    return fig, ax


def run_computation(parameters, storage=None):
    _nameExp = parameters["geometry"]["geom_type"]
    lc = parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"]

    # Get geometry model
    parameters["geometry"]["geom_type"]

    Lx = parameters.get("geometry").get("Lx", 1.0)
    Ly = parameters.get("geometry").get("Ly", 0.1)

    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([Lx, Ly])],
        [int(Lx / lc), int(Ly / lc)],
        cell_type=dolfinx.mesh.CellType.triangle,
    )
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    outdir = os.path.join(os.path.dirname(__file__), "output")

    prefix = setup_output_directory(storage, parameters, outdir)

    signature = save_parameters(parameters, prefix)

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

    _V_ux = V_u.sub(0)
    V_ux, V_ux_to_V_u = _V_ux.collapse()

    u_t = Function(V_u, name="InelasticDisplacement")
    u_xt = Function(V_ux, name="BoundaryDisplacement")

    # Bounds
    alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")

    # Define the state
    # zero_u = Function(V_u, name="BoundaryDatum")
    # zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))

    tau = Constant(mesh, np.array(0.0, dtype=PETSc.ScalarType))

    u_xt.interpolate(lambda x: 2.0 * tau * (x[0] - Lx / 2.0) / Lx)
    eps_t = tau * ufl.as_tensor([[1.0, 0], [0, 0.0]])
    eps_0 = ufl.as_tensor([[0.0, 0], [0, 0.0]])

    with dolfinx.common.Timer("~Debug"):
        from irrevolutions.utils.viz import plot_profile

        plotter = pyvista.Plotter(
            title="Test bcs",
            window_size=[800, 600],
            shape=(1, 1),
        )
        tol = 1e-3
        xs = np.linspace(0 + tol, Lx - tol, 101)
        points = np.zeros((3, 101))
        points[0] = xs

        profile, data = plot_profile(
            u_xt,
            points,
            plotter,
            subplotnumber=1,
            lineproperties={"c": "k", "label": "$u_tx$"},
        )
        profile.savefig(f"{prefix}/profile.png")

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    def ux_boundary(x):
        return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], Lx))

    ux_boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, ux_boundary
    )
    ux_boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V_u.sub(0), mesh.topology.dim - 1, ux_boundary_facets
    )

    # ux_boundary_dofs = dolfinx.fem.locate_dofs_geometrical(
    #     V_u.sub(0).collapse()[0],
    #     lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], Lx)),
    # )

    # u_boundary_dofs = dolfinx.fem.locate_dofs_topological(V_u, fdim, boundary_facets)
    # ux_boundary_dofs = dolfinx.fem.locate_dofs_topological(V_ux, fdim, boundary_facets)
    # ux_boundary_dofs = dolfinx.fem.locate_dofs_topological(
    #     V_u.sub(0), fdim, boundary_facets
    # )

    for f in [u, u_t, u_xt, alpha_lb, alpha_ub]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    bcs_u = [dirichletbc(u_xt, ux_boundary_dofs)]
    # __import__("pdb").set_trace()
    # bcs_u = [dirichletbc(u_xt, ux_boundary_dofs, V_u.sub(0))]
    # bcs_u = [dirichletbc(u_xt, ux_boundary_dofs)]
    # bcs_u = [dirichletbc(u_xt, u_boundary_dofs)]
    # bcs_u = [dirichletbc(u_t, u_boundary_dofs)]
    bcs_u = []
    bcs_alpha = []
    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    dx = ufl.Measure("dx", domain=mesh)

    # model = ThinFilm(parameters["model"])
    model = ThinFilm(parameters["model"], eps_0=eps_t)
    total_energy = model.total_energy_density(state) * dx
    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

    hybrid = HybridSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )
    hybrid.damage.solver.setMonitor(None)
    hybrid.elasticity.solver.setMonitor(None)

    bifurcation = BifurcationSolver(
        total_energy, state, bcs, bifurcation_parameters=parameters.get("stability")
    )

    stability = StabilitySolver(
        total_energy, state, bcs, cone_parameters=parameters.get("stability")
    )

    linesearch = LineSearch(
        total_energy,
        state,
        linesearch_parameters=parameters.get("stability").get("linesearch"),
    )

    iterator = StabilityStepper(loads)

    for i_t, t in enumerate(loads):
        tau.value = t

        # update the lower bound
        alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
        alpha_lb.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        _logger.critical(f"-- Solving Equilibrium (Criticality) for t = {t:3.2f} --")
        hybrid.solve(alpha_lb)

        _logger.critical(f"-- Solving Bifurcation (Uniqueness) for t = {t:3.2f} --")
        is_unique = bifurcation.solve(alpha_lb)
        is_elastic = not bifurcation._is_critical(alpha_lb)
        inertia = bifurcation.get_inertia()

        ColorPrint.print_bold(f"State is elastic: {is_elastic}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        ColorPrint.print_bold(f"Evolution is unique: {is_unique}")

        z0 = (
            bifurcation._spectrum[0]["xk"]
            if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
            else None
        )

        _logger.critical(f"-- Solving Stability (Stability) for t = {t:3.2f} --")
        stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)

        with dolfinx.common.Timer(f"~Output and Storage") as timer:
            if BINARY_DATA:
                with XDMFFile(
                    comm,
                    f"{prefix}/{_nameExp}.xdmf",
                    "a",
                    encoding=XDMFFile.Encoding.HDF5,
                ) as file:
                    file.write_function(u, t)
                    file.write_function(alpha, t)

        with dolfinx.common.Timer("~Postprocessing and Vis"):
            fracture_energy = comm.allreduce(
                assemble_scalar(form(model.damage_energy_density(state) * dx)),
                op=MPI.SUM,
            )
            elastic_energy = comm.allreduce(
                assemble_scalar(form(model.elastic_energy_density(state) * dx)),
                op=MPI.SUM,
            )

        history_data["elastic_energy"].append(elastic_energy)
        history_data["fracture_energy"].append(fracture_energy)
        history_data["total_energy"].append(elastic_energy + fracture_energy)
        history_data["load"].append(t)
        history_data["unique"].append(is_unique)
        history_data["stable"].append(stable)
        history_data["inertia"].append(inertia)
        history_data["eigs_cone"].append(stability.solution["lambda_t"])
        history_data["eigs_ball"].append(bifurcation.data["eigs"])
        history_data["equilibrium_data"].append(hybrid.data)
        history_data["cone_data"].append(stability.data)

        with dolfinx.common.Timer("~Postprocessing and Vis"):
            fig, ax = plot_energies(history_data, _storage)
            fig.savefig(f"{_storage}/energies.png")
            plt.close(fig)

            fig, ax = plot_spectrum(history_data)
            fig.savefig(f"{_storage}/spectrum.png")
            plt.close(fig)

    return history_data


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
    parameters["geometry"]["Lx"] = 2.0
    parameters["geometry"]["Ly"] = 0.1

    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = 0.02
    parameters["model"]["ell_e"] = 0.1

    parameters["loading"]["min"] = 0.99
    parameters["loading"]["max"] = 1.5
    parameters["loading"]["steps"] = 100

    parameters["solvers"]["damage"]["snes"]["snes_monitor"] = None
    parameters["solvers"]["elasticity"]["snes"]["snes_monitor"] = None
    parameters["solvers"]["newton"]["snes_monitor"] = None

    parameters["solvers"]["damage_elasticity"]["max_it"] = 1000
    parameters["solvers"]["damage_elasticity"]["alpha_rtol"] = 1e-4

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files

if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # with pkg_resources.path("crunchy.test", "parameters.yml") as f:
    #     print(f"Parameters file: {f}")

    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yaml"),
        model="at1",
    )

    # Run computation
    # _storage = f"output/MPI-{MPI.COMM_WORLD.Get_size()}/{signature[0:6]}"
    _storage = f"output/MPI-{MPI.COMM_WORLD.Get_size()}/test"
    visualization = Visualization(_storage)

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data = run_computation(parameters, _storage)

    experimental_data = pd.DataFrame(history_data)
    print(experimental_data)

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
