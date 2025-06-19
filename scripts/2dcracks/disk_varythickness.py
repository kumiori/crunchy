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
import basix

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

from crunchy.mesh import (
    mesh_circle_with_holes_gmshapi_old,
    mesh_circle_with_holes_gmshapi,
)
from irrevolutions.utils.plots import (
    plot_energies,
)

from crunchy.utils import plot_spectrum
from matplotlib.colors import LinearSegmentedColormap
from crunchy.utils import plot_spectrum, save_stress_components
from irrevolutions.models import BanquiseVaryingThickness as VariableThickness


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

    # msh_file = f"meshes/meshwithholes-{geom_signature}.msh"
    # if not os.path.exists(msh_file):
    # gmsh_model, tdim = mesh_circle_with_holes_gmshapi(
    #     "discwithholes",
    #     parameters["geometry"]["R"],
    #     lc=lc,
    #     tdim=2,
    #     num_holes=parameters["geometry"]["num_holes"],
    #     hole_radius=parameters["geometry"]["hole_radius"],
    #     hole_positions=None,
    #     refinement_factor=0.8,
    #     order=1,
    #     msh_file=msh_file,
    #     comm=MPI.COMM_WORLD,
    # )

    # mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    #     msh_file, gdim=2, comm=MPI.COMM_WORLD
    # )

    gmsh_model, tdim = mesh_circle_gmshapi(
        _nameExp, R, lc, tdim=2, order=1, msh_file=None, comm=MPI.COMM_WORLD
    )
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as xdmf:
        xdmf.write_mesh(mesh)

    stress_output_file = f"{prefix}/stress.xdmf"
    with XDMFFile(
        mesh.comm, stress_output_file, "w", encoding=XDMFFile.Encoding.HDF5
    ) as xdmf:
        xdmf.write_mesh(mesh)

    if comm.rank == 0:
        plt.figure()
        ax = plot_mesh(mesh)
        fig = ax.get_figure()
        fig.savefig(f"{prefix}/mesh.png")
        # fig.savefig(f"{msh_file[0:-3]}.png")

    # Functional Setting
    # gmsh_model, tdim = mesh_circle_gmshapi(
    #     _nameExp, R, lc, tdim=2, order=1, msh_file=None, comm=MPI.COMM_WORLD
    # )
    # mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    V_u, V_alpha = create_function_spaces_2d(mesh)
    W = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 0))
    u, u_, alpha, β, v, state = initialise_functions(V_u, V_alpha)

    # Bounds
    alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")
    thickness = dolfinx.fem.Function(V_alpha)

    # Get the bounds of the domain
    x_coords = mesh.geometry.x[:, 0]  # Extract x[0] values
    x_min, x_max = np.min(x_coords), np.max(x_coords)

    # Define the linear thickness function
    def linear_thickness(x):
        return 0.5 + (x[0] - x_min) / (x_max - x_min) * 1.0

    # Interpolate the thickness function into the scalar field
    thickness.interpolate(linear_thickness)

    def radial_field(x):
        # r = np.sqrt(x[0]**2 + x[1]**2)
        u_x = x[0]
        u_y = x[1]
        return np.array([u_x, u_y])

    tau = Constant(mesh, np.array(0.0, dtype=PETSc.ScalarType))
    u_t = Function(V_u, name="InelasticDisplacement")

    u_t.interpolate(lambda x: radial_field(x) * tau)
    eps_t = tau * ufl.as_tensor([[1.0, 0], [0, 1.0]])

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    for f in [u, u_t, alpha_lb, alpha_ub]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    # boundary_dofs = dolfinx.fem.locate_dofs_topological(
    #     V_u, mesh.topology.dim - 1, facet_tags.indices
    # )
    mesh_boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    # ext_boundary_u_dofs = dolfinx.fem.locate_dofs_topological(
    #     V_u, fdim, mesh_boundary_facets
    # )

    # holes_bc_values = dolfinx.fem.Constant(mesh, PETSc.ScalarType([0.0, 0.0]))
    # subtract the holes from the boundary dofs
    # outer_ext_dofs = np.setdiff1d(ext_boundary_u_dofs, boundary_dofs)

    # locate boundary dofs geometrically
    u_boundary_dofs = dolfinx.fem.locate_dofs_geometrical(
        V_u, lambda x: np.isclose(x[0] ** 2 + x[1] ** 2, R**2)
    )
    # __import__("pdb").set_trace()
    # bcs_u = [
    # dolfinx.fem.dirichletbc(holes_bc_values, boundary_dofs, V_u),
    # dolfinx.fem.dirichletbc(u_t, u_boundary_dofs),
    # ]

    bcs_u = []
    bcs_alpha = []

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    dx = ufl.Measure("dx", domain=mesh)

    model = ThinFilm(parameters["model"], eps_0=eps_t)
    # model = VariableThickness(thickness, parameters["model"], eps_0=eps_t)

    total_energy = model.total_energy_density(state) * dx
    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

    equilibrium = HybridSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )
    equilibrium.damage.solver.setMonitor(None)
    equilibrium.elasticity.solver.setMonitor(None)

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
    history_data["average_stress"] = []

    while True:
        try:
            i_t = next(iterator) - 1
        except StopIteration:
            break

        # Update loading and lower bound
        # eps_t.value = loads[i_t]
        # tau.value = loads[i_t]
        # tau.value = 0
        u_t.interpolate(lambda x: radial_field(x) * tau)

        # u_zero.interpolate(lambda x: eps_t / 2.0 * (2 * x[0] - Lx))
        # u_zero.x.petsc_vec.ghostUpdate(
        # addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        # )

        alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
        alpha_lb.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Solve for equilibrium
        _logger.critical(f"-- Solving for t = {loads[i_t]:3.2f} --")
        with dolfinx.common.Timer(f"~First Order: Equilibrium") as timer:
            equilibrium.solve(alpha_lb)

        # Solve for bifurcation
        is_unique = bifurcation.solve(alpha_lb)
        is_elastic = not bifurcation._is_critical(alpha_lb)
        inertia = bifurcation.get_inertia()

        z0 = (
            bifurcation._spectrum[0]["xk"]
            if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
            else None
        )

        # Solve for stability
        stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)

        # Log results
        _logger.info(f"Stability of state at load {loads[i_t]:.2f}: {stable}")
        ColorPrint.print_bold(f"Evolution is unique: {is_unique}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        ColorPrint.print_bold(f"State's stable: {stable}")

        # Postprocess energies
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
        history_data["load"].append(loads[i_t])
        history_data["unique"].append(is_unique)
        history_data["stable"].append(stable)
        history_data["inertia"].append(inertia)
        history_data["eigs_cone"].append(stability.solution["lambda_t"])
        history_data["eigs_ball"].append(bifurcation.data["eigs"])
        history_data["equilibrium_data"].append(equilibrium.data)
        history_data["cone_data"].append(stability.data)

        # Visualize state
        with dolfinx.common.Timer("~Visualisation") as timer:
            fig, ax = plot_energies(history_data, _storage)
            fig.savefig(f"{_storage}/energies.png")
            plt.close(fig)

            fig, ax = plot_spectrum(history_data)
            fig.savefig(f"{_storage}/spectrum-Λ={parameters['model']['ell_e']}.png")
            plt.close(fig)

            pyvista.OFF_SCREEN = True
            plotter = pyvista.Plotter(
                title="State Evolution",
                window_size=[1600, 600],
                shape=(1, 2),
            )

            # Plot alpha
            _plt, grid = plot_scalar(
                alpha,
                plotter,
                subplot=(0, 0),
                lineproperties={"cmap": "gray_r", "clim": (0.0, 1.0)},
            )

            # Add boundary lines
            boundary_edges = grid.extract_feature_edges(
                boundary_edges=True, feature_edges=False
            )
            _plt.add_mesh(
                boundary_edges, style="wireframe", color="black", line_width=3.0
            )

            # Plot displacement
            _plt = plot_vector(u, plotter, subplot=(0, 1))
            _plt.screenshot(f"{_storage}/state_t{i_t:03}.png")

        # Save stress components
        hydrostatic, deviatoric, stress_tensor = save_stress_components(
            mesh, model.stress(model.eps(u), alpha), f"{_storage}/stress", t=loads[i_t]
        )

        average_stress = model.stress_average(model.eps(u), alpha)
        history_data["average_stress"].append(average_stress)

        # Handle bifurcation if unstable
        if not stable:
            iterator.pause_time()
            perturbation = {"v": v, "beta": β}
            interval = linesearch.get_unilateral_interval(state, perturbation)

            h_opt, _, _, _ = linesearch.search(state, perturbation, interval, m=4)
            linesearch.perturb(state, perturbation, h_opt)

            # Postprocess perturbed state
            # fracture_energy, elastic_energy = postprocess(
            #     parameters,
            #     _nameExp,
            #     prefix,
            #     v,
            #     β,
            #     state,
            #     u_zero,
            #     dx,
            #     bifurcation,
            #     stability,
            #     i_t,
            #     model=model,
            # )

        # Save results
        # with XDMFFile(
        # comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        # ) as file:
        # file.write_function(u, loads[i_t])
        # file.write_function(alpha, loads[i_t])

    # Save experimental data
    __import__("pdb").set_trace()
    experimental_data = pd.DataFrame(history_data)
    if comm.rank == 0:
        experimental_data.to_pickle(f"{_storage}/experimental_data.pkl")


def load_parameters(file_path, model="at1"):
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["geometry"]["R"] = 1.0
    # parameters["geometry"]["num_holes"] = 3
    # parameters["geometry"]["hole_radius"] = 0.005

    parameters["geometry"]["mesh_size_factor"] = 2
    parameters["model"]["ell"] = 0.05
    parameters["model"]["ell_e"] = 0.3

    parameters["loading"]["min"] = 0.0
    parameters["loading"]["max"] = 0.1
    parameters["loading"]["steps"] = 2

    parameters["solvers"]["damage"]["snes"]["snes_monitor"] = None
    parameters["solvers"]["elasticity"]["snes"]["snes_monitor"] = None
    parameters["solvers"]["newton"]["snes_monitor"] = None

    parameters["solvers"]["damage_elasticity"]["max_it"] = 2000
    parameters["solvers"]["damage_elasticity"]["alpha_rtol"] = 1e-4

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    return parameters, signature


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)
    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yaml"),
    )
    # _storage = f"output/varythickness/MPI-{MPI.COMM_WORLD.Get_size()}/{signature[0:6]}"
    _storage = f"output/varythickness/MPI-{MPI.COMM_WORLD.Get_size()}/test/"
    visualization = Visualization(_storage)

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data = run_computation(parameters, _storage)

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
