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


def split_stress(stress_tensor):
    """
    Split stress tensor into hydrostatic and deviatoric components.

    Parameters:
    stress_tensor (ufl.Expr): The stress tensor to split.

    Returns:
    tuple: hydrostatic and deviatoric stress tensors.
    """
    # Identity tensor
    I = ufl.Identity(len(stress_tensor.ufl_shape))

    # Compute hydrostatic stress
    sigma_hydrostatic = ufl.tr(stress_tensor) / len(stress_tensor.ufl_shape)
    # Compute deviatoric stress
    sigma_deviatoric = stress_tensor - sigma_hydrostatic * I

    return sigma_hydrostatic, sigma_deviatoric


def save_stress_components(mesh, stress_tensor, output_file, t=0):
    """
    Save stress tensor components, hydrostatic, and deviatoric parts.

    Parameters:
    mesh (dolfinx.Mesh): Mesh object.
    stress_tensor (ufl.Expr): Stress tensor to process.
    output_file (str): Path to the XDMF file for saving data.
    """
    dtype = PETSc.ScalarType
    # Create function spaces
    scalar_space = dolfinx.fem.functionspace(
        mesh, ("Discontinuous Lagrange", 0)
    )  # For scalars
    element_vec = basix.ufl.element(
        "Discontinuous Lagrange", mesh.basix_cell(), degree=0, shape=(2,)
    )
    element_vec_3d = basix.ufl.element(
        "Discontinuous Lagrange", mesh.basix_cell(), degree=0, shape=(3,)
    )
    element_tens = basix.ufl.element(
        "Discontinuous Lagrange", mesh.basix_cell(), degree=0, shape=(2, 2)
    )
    vector_space = dolfinx.fem.functionspace(mesh, element_vec)
    stress_vector_space = dolfinx.fem.functionspace(mesh, element_vec_3d)
    tensor_space = dolfinx.fem.functionspace(mesh, element_tens)
    # )  # For stress components

    # Create functions for storage
    stress_components = Function(vector_space, name="StressComponents")
    hydrostatic = Function(scalar_space, name="HydrostaticStress")
    deviatoric_components = Function(vector_space, name="DeviatoricStressComponents")
    deviatoric = Function(tensor_space, name="DeviatoricStress")
    stress_vector_function = dolfinx.fem.Function(
        stress_vector_space, name="StressVector"
    )

    # Extract components
    sigma_hydrostatic, sigma_deviatoric = split_stress(stress_tensor)
    sigma_hydro_expr = dolfinx.fem.Expression(
        sigma_hydrostatic, scalar_space.element.interpolation_points(), dtype=dtype
    )
    stress_vector_function.interpolate(
        dolfinx.fem.Expression(
            ufl.as_vector(
                [stress_tensor[0, 0], stress_tensor[1, 1], stress_tensor[0, 1]]
            ),
            stress_vector_space.element.interpolation_points(),
            dtype=PETSc.ScalarType,
        )
    )
    sigma_dev_expr = dolfinx.fem.Expression(
        sigma_deviatoric, vector_space.element.interpolation_points(), dtype=dtype
    )

    # Interpolate components
    hydrostatic.interpolate(sigma_hydro_expr)
    deviatoric.interpolate(sigma_dev_expr)

    # Save to XDMF
    with XDMFFile(mesh.comm, output_file, "a") as xdmf:
        # xdmf.write_mesh(mesh)
        xdmf.write_function(hydrostatic, t)
        xdmf.write_function(stress_vector_function, t)
        xdmf.write_function(deviatoric, t)

    return hydrostatic, stress_vector_function, deviatoric


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

    msh_file = f"meshes/meshwithholes-{geom_signature}.msh"
    # if not os.path.exists(msh_file):
    gmsh_model, tdim = mesh_circle_with_holes_gmshapi(
        "discwithholes",
        parameters["geometry"]["R"],
        lc=lc,
        tdim=2,
        num_holes=parameters["geometry"]["num_holes"],
        hole_radius=parameters["geometry"]["hole_radius"],
        hole_positions=None,
        refinement_factor=0.8,
        order=1,
        msh_file=msh_file,
        comm=MPI.COMM_WORLD,
    )

    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        msh_file, gdim=2, comm=MPI.COMM_WORLD
    )

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    stress_output_file = f"{prefix}/stress.xdmf"
    # Save to XDMF
    with XDMFFile(mesh.comm, stress_output_file, "w") as xdmf:
        xdmf.write_mesh(mesh)

    if comm.rank == 0:
        plt.figure()
        ax = plot_mesh(mesh)
        fig = ax.get_figure()
        fig.savefig(f"{prefix}/mesh.png")
        fig.savefig(f"{msh_file[0:-3]}.png")

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

    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V_u, mesh.topology.dim - 1, facet_tags.indices
    )
    mesh_boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    ext_boundary_u_dofs = dolfinx.fem.locate_dofs_topological(
        V_u, fdim, mesh_boundary_facets
    )

    holes_bc_values = dolfinx.fem.Constant(mesh, PETSc.ScalarType([0.0, 0.0]))
    # subtract the holes from the boundary dofs
    # outer_ext_dofs = np.setdiff1d(ext_boundary_u_dofs, boundary_dofs)

    # locate boundary dofs geometrically
    u_boundary_dofs = dolfinx.fem.locate_dofs_geometrical(
        V_u, lambda x: np.isclose(x[0] ** 2 + x[1] ** 2, R**2)
    )
    # __import__("pdb").set_trace()
    bcs_u = [
        dolfinx.fem.dirichletbc(holes_bc_values, boundary_dofs, V_u),
        # dolfinx.fem.dirichletbc(u_t, u_boundary_dofs),
    ]

    # bcs_u = []
    bcs_alpha = []

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    dx = ufl.Measure("dx", domain=mesh)

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
    history_data["average_stress"] = []

    for i_t, t in enumerate(loads):
        tau.value = t
        u_t.interpolate(lambda x: radial_field(x) * tau)

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
            fig.savefig(f"{_storage}/spectrum-Λ={parameters['model']['ell_e']}.png")
            plt.close(fig)

            # xvfb.start_xvfb(wait=0.05)
            pyvista.OFF_SCREEN = True

            plotter = pyvista.Plotter(
                title="Displacement",
                window_size=[700, 1500],
                # window_size=[1600, 900],
                shape=(2, 1),
            )

            # Create a custom colormap: transparent for alpha <= 0.7, black for alpha > 0.7
            colors = [(0, 0, 0, 0.05), (0, 0, 0, 1)]  # RGBA: transparent to black
            custom_cmap = LinearSegmentedColormap.from_list(
                "transparent_to_black", colors
            )

            lineproperties_alpha = {
                "scalar_bar_args": {
                    "title": "Damage",  # Label for the scalar bar
                    "title_font_size": 22,  # Font size for the title
                    "label_font_size": 18,  # Font size for the scalar bar labels
                    "vertical": False,  # Vertical orientation of the scalar bar
                    "position_y": 0.025,  # Adjust vertical position
                    "position_x": 0.6,  # Adjust horizontal position
                    "height": 0.03,  # Height of the scalar bar
                    "width": 0.3,  # Width of the scalar bar
                    "n_labels": 3,  # Adjust number of ticks
                    "fmt": "%.1f",  # Format for two decimal places
                },
                "cmap": custom_cmap,  # Colormap to use
                # "cmap": "gray_r",  # Colormap to use
                # "clim": (0.0, 1.0),  # vmin and vmax for the color scaling
                "clim": (0.0, 0.9),  # vmin and vmax for the color scaling
                "above_color": "red",
                "show_edges": False,  # Optional: Hide edges for smoother appearance
                "opacity": 1.0,  # Optional: Set opacity (1.0 for fully opaque)
                "annotations": {
                    0: "Sound",
                    0.9: "Fracture",
                },
            }
            lineproperties_u = {
                "scalar_bar_args": {
                    "title": "Displacement",  # Label for the scalar bar
                    "title_font_size": 22,  # Font size for the title
                    "label_font_size": 18,  # Font size for the scalar bar labels
                    "vertical": False,  # Vertical orientation of the scalar bar
                    "position_y": 0.025,  # Adjust vertical position
                    "position_x": 0.6,  # Adjust horizontal position
                    "height": 0.03,  # Height of the scalar bar
                    "width": 0.3,  # Width of the scalar bar
                    "n_labels": 4,  # Adjust number of ticks
                    "fmt": "%.2f",  # Format for two decimal places
                },
                # "cmap": custom_cmap,  # Colormap to use
                # "cmap": "gray_r",  # Colormap to use
                "show_edges": False,  # Optional: Hide edges for smoother appearance
                "opacity": 1.0,  # Optional: Set opacity (1.0 for fully opaque)
            }
            _plt, grid = plot_scalar(
                alpha, plotter, subplot=(0, 0), lineproperties=lineproperties_alpha
            )

            boundary_edges = grid.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                non_manifold_edges=False,
                manifold_edges=False,
            )
            boundary_line_properties = {
                "color": "black",  # Set the line color to red
                "line_width": 3.0,  # Line thickness
                "opacity": 1.0,  # Fully opaque
            }
            _plt.add_mesh(boundary_edges, style="wireframe", **boundary_line_properties)
            plotter.camera_position = "xy"
            plotter.camera.zoom(1.3)
            _plt = plot_vector(
                u, plotter, subplot=(1, 0), lineproperties=lineproperties_u
            )
            _plt.add_mesh(boundary_edges, style="wireframe", **boundary_line_properties)
            plotter.camera_position = "xy"
            plotter.camera.zoom(1.3)
            _plt.screenshot(os.path.join(prefix, f"state_MPI{comm.size}-{i_t}.png"))

            average_stress = model.stress_average(model.eps(u), alpha)
            stress = model.stress(model.eps(u), alpha)
            stress_projected = ufl.as_tensor(
                [
                    [stress[0, 0], stress[0, 1]],
                    [stress[1, 0], stress[1, 1]],
                ]
            )
            history_data["average_stress"].append(average_stress)

            hydrostatic, stress_vector_function, deviatoric = save_stress_components(
                mesh, stress_projected, stress_output_file, t
            )

        with dolfinx.common.Timer(f"~Output and Storage") as timer:
            with XDMFFile(
                comm,
                f"{prefix}/{_nameExp}.xdmf",
                "a",
                encoding=XDMFFile.Encoding.HDF5,
            ) as file:
                file.write_function(u, t)
                file.write_function(alpha, t)
                # file.write_function(stress_h, t)

        experimental_data = pd.DataFrame(history_data)

        if comm.rank == 0:
            experimental_data.to_pickle(f"{_storage}/experimental_data.pkl")

    return history_data


def load_parameters(file_path, model="at1"):
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["geometry"]["R"] = 1.0
    parameters["geometry"]["num_holes"] = 7
    parameters["geometry"]["hole_radius"] = 0.002

    parameters["geometry"]["mesh_size_factor"] = 3
    parameters["model"]["ell"] = 0.05
    parameters["model"]["ell_e"] = 0.2

    parameters["loading"]["min"] = 0.1
    parameters["loading"]["max"] = 1.1
    parameters["loading"]["steps"] = 30

    parameters["solvers"]["damage"]["snes"]["snes_monitor"] = None
    parameters["solvers"]["elasticity"]["snes"]["snes_monitor"] = None
    parameters["solvers"]["newton"]["snes_monitor"] = None

    parameters["solvers"]["damage_elasticity"]["max_it"] = 4000
    parameters["solvers"]["damage_elasticity"]["alpha_rtol"] = 1e-5

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    return parameters, signature


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)
    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yaml"),
    )
    _storage = f"output/with_holes/MPI-{MPI.COMM_WORLD.Get_size()}/{signature[0:6]}"
    visualization = Visualization(_storage)
    ColorPrint.print_bold(f"===================- {_storage} -=================")

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
