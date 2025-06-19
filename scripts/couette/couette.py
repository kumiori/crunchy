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
from crunchy.mesh import mesh_circle_with_holes_gmshapi
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
from irrevolutions.utils import _logger, setup_logger_mpi, history_data
import irrevolutions.models as models
from matplotlib import cm

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

from dolfinx.plot import vtk_mesh as compute_topology

from crunchy.utils import (
    positive_negative_trace,
    split_stress,
)
from crunchy.utils import write_history_data

from ufl import as_vector, as_matrix, cos, sin


def linear_thickness(x, R_in, R_out=1.0, H_max=2, H_min=0.5):
    """
    Compute the thickness at points (x, y) via linear interpolation.

    The thickness is set to H_max for any point with radial distance r <= R_in,
    and to H_min for any point with r >= R_out. For points with R_in < r < R_out,
    the thickness is interpolated linearly.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the points.
    y : np.ndarray
        The y-coordinates of the points.
    R_in : float
        The inner radius at which the thickness is H_max.
    R_out : float
        The outer radius at which the thickness is H_min.
    H_max : float
        The maximum thickness at r <= R_in.
    H_min : float
        The minimum thickness at r >= R_out.

    Returns
    -------
    thickness : np.ndarray
        An array of thickness values corresponding to each (x, y) point.
    """
    # Compute the radial coordinate r = sqrt(x^2 + y^2)

    r = np.sqrt(x[0] ** 2 + x[1] ** 2)

    # Create the thickness array using a piecewise definition:
    #   If r <= R_in: assign H_max.
    #   If r >= R_out: assign H_min.
    #   Otherwise, linearly interpolate between H_max and H_min.
    thickness = np.where(
        r <= R_in,
        H_max,
        np.where(
            r >= R_out, H_min, H_min + (H_max - H_min) * ((R_out - r) / (R_out - R_in))
        ),
    )
    return thickness


def rotation_displacement(x, centre, angle):
    """
    Compute a rotation-induced displacement field around a given centre.

    The function assumes x has shape (d, N) where d>=2. Only the first two
    components are used for the rotation.

    Parameters
    ----------
    x : np.ndarray
        Array of coordinates with shape (d, N).
    centre : tuple of floats
        The (cx, cy) coordinates about which to rotate.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    u : np.ndarray
        Displacement field of shape (d, N). Only the first two components are nonzero.
    """
    # Ensure x is a NumPy array
    x = x[:2]
    x = np.asarray(x)  # Expected shape: (2, N)
    N = x.shape[1]

    # Unpack centre coordinates and create centre array with shape (2, N)
    cx, cy = centre
    centre_arr = np.array([[cx], [cy]])  # shape (2, 1)
    centre_arr = np.repeat(centre_arr, N, axis=1)  # shape (2, N)

    # Compute the vector from centre to each point
    xi = x - centre_arr  # shape (2, N)
    # Build the rotation matrix explicitly
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s], [s, c]])  # shape (2,2)

    # Compute the rotated coordinates
    rotated = R @ xi  # shape (2, N)
    return rotated - xi


def postprocess(
    parameters,
    _nameExp,
    prefix,
    β,
    v,
    state,
    t,
    dx,
    loads,
    equilibrium,
    bifurcation,
    stability,
    inertia,
    i_t,
    model,
    history_data,
    signature="",
):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        fracture_energy = comm.allreduce(
            assemble_scalar(form(model.damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(model.elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )

        write_history_data(
            equilibrium=equilibrium,
            bifurcation=bifurcation,
            stability=stability,
            history_data=history_data,
            t=loads[i_t],
            inertia=inertia,
            stable=np.nan,
            energies=[elastic_energy, fracture_energy],
        )

        alpha, u = state["alpha"], state["u"]
        mesh = alpha.function_space.mesh
        with XDMFFile(
            comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        ) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)

        stress_tensor = model.stress(model.eps(u), alpha)

        sigma_hydrostatic, sigma_deviatoric = split_stress(stress_tensor)
        tr_plus, tr_minus = positive_negative_trace(model.eps(u))
        tr_strain = ufl.tr(model.eps(u))

        dtype = PETSc.ScalarType

        # Create function spaces for scalar and vector quantities
        scalar_space = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 0))
        vector_space = dolfinx.fem.functionspace(
            mesh, ("Discontinuous Lagrange", 0, (2,))
        )
        tensor_space = dolfinx.fem.functionspace(
            mesh, ("Discontinuous Lagrange", 0, (2, 2))
        )

        # Create functions for storage
        stress_deviatoric = dolfinx.fem.Function(tensor_space, name="DeviatoricStress")
        stress_hydrostatic = dolfinx.fem.Function(
            scalar_space, name="HydrostaticStress"
        )
        strain_positive = dolfinx.fem.Function(
            scalar_space, name="StrainPositiveVolumetric"
        )
        strain_negative = dolfinx.fem.Function(
            scalar_space, name="StrainNegativeVolumetric"
        )
        trace_strain = dolfinx.fem.Function(scalar_space, name="TraceStrain")

        # Interpolate computed quantities
        stress_deviatoric.interpolate(
            dolfinx.fem.Expression(
                sigma_deviatoric,
                vector_space.element.interpolation_points(),
                dtype=dtype,
            )
        )
        stress_hydrostatic.interpolate(
            dolfinx.fem.Expression(
                sigma_hydrostatic,
                scalar_space.element.interpolation_points(),
                dtype=dtype,
            )
        )
        strain_positive.interpolate(
            dolfinx.fem.Expression(
                tr_plus, scalar_space.element.interpolation_points(), dtype=dtype
            )
        )
        strain_negative.interpolate(
            dolfinx.fem.Expression(
                tr_minus, scalar_space.element.interpolation_points(), dtype=dtype
            )
        )
        trace_strain.interpolate(
            dolfinx.fem.Expression(
                tr_strain, scalar_space.element.interpolation_points(), dtype=dtype
            )
        )

        with dolfinx.common.Timer("~Visualisation") as timer:
            fig, ax = plot_energies(history_data, _storage)
            fig.savefig(f"{_storage}/energies.png")
            plt.close(fig)

            pyvista.OFF_SCREEN = True
            topology, cell_types, geometry = compute_topology(mesh, mesh.topology.dim)
            # topology, cell_types, geometry = create_vtk_mesh(mesh, mesh.topology.dim)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            num_cells = len(grid.celltypes)  # This should match DG0 function size
            cmap_positive = cm.get_cmap("coolwarm").copy()
            cmap_negative = cm.get_cmap("coolwarm").copy()
            grid_positive = grid.copy()
            grid_negative = grid.copy()
            # Add computed stress components to the grid
            grid.cell_data["Deviatoric Stress"] = stress_deviatoric.x.array.real[
                :num_cells
            ]
            grid.cell_data["Hydrostatic Stress"] = stress_hydrostatic.x.array.real[
                :num_cells
            ]
            grid_positive.cell_data["Positive Strain"] = strain_positive.x.array.real[
                :num_cells
            ]
            grid_negative.cell_data["Negative Strain"] = strain_negative.x.array.real[
                :num_cells
            ]
            grid.cell_data["Trace Strain"] = trace_strain.x.array.real[:num_cells]

            grid.compute_cell_sizes(length=False, volume=False)
            grid_point = grid.cell_data_to_point_data()

            # Create PyVista plotter
            plotter = pyvista.Plotter(shape=(2, 2))

            # Define common colormap
            colormap = "coolwarm"

            # Plot each component
            plotter.subplot(0, 0)
            plotter.add_mesh(grid, scalars="Deviatoric Stress", cmap=colormap)
            plotter.add_text("Deviatoric Stress")
            plotter.view_xy()

            plotter.subplot(0, 1)
            plotter.add_mesh(grid, scalars="Hydrostatic Stress", cmap=colormap)
            plotter.add_text("Hydrostatic Stress")
            plotter.view_xy()

            contours = grid_point.contour([0], scalars="Trace Strain")
            plotter.subplot(1, 0)
            vmin = min(grid_negative.cell_data["Negative Strain"])
            vmax = max(grid_positive.cell_data["Positive Strain"])
            plotter.add_mesh(
                grid_positive,
                scalars="Positive Strain",
                cmap=cmap_positive,
                clim=(0, vmax),
            )
            if len(contours.points) > 0:
                plotter.add_mesh(contours, color="white", line_width=3)

            plotter.add_text("Positive Volumetric Strain")
            plotter.view_xy()

            plotter.subplot(1, 1)
            plotter.add_mesh(
                grid_negative,
                scalars="Negative Strain",
                cmap=cmap_negative,
                clim=(vmin, 0),
            )
            if len(contours.points) > 0:
                plotter.add_mesh(contours, color="white", line_width=3)
            plotter.add_text("Negative Volumetric Strain")
            plotter.view_xy()

            _R = parameters["geometry"]["R"]
            _ell = parameters["model"]["ell"]

            line_radius = pyvista.Line(pointa=(0, 0, 0), pointb=(_R, 0, 0))
            plotter.add_mesh(line_radius, color="black", line_width=4)
            plotter.add_text(f"Radius = {_R}", position="lower_left", font_size=14)
            line_ell = pyvista.Line(
                pointa=(0, -_ell * 2, 0), pointb=(_ell, -_ell * 2, 0)
            )
            plotter.add_mesh(line_ell, color="blue", line_width=4)
            plotter.add_text(f"$\\ell$ = {_ell}", position="lower_right", font_size=14)

            plotter.screenshot(f"{_storage}/stess_state_t{i_t:03}.png")
            plotter.close()  # Properly close the render window

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

            # Plot displacement
            _plt, grid = plot_vector(u, plotter, subplot=(0, 1))
            _plt.screenshot(f"{_storage}/state_t{i_t:03}.png")

            try:
                fig, ax = plot_spectrum(history_data)
                fig.savefig(f"{_storage}/spectrum.png")
                plt.close(fig)
            except Exception as e:
                _logger.error(f"Error plotting spectrum: {e}")

            plotter.clear()  # Remove all actors and settings
            plotter.close()  # Properly close the render window

    return fracture_energy, elastic_energy


def run_computation(parameters, storage):
    nameExp = parameters["geometry"]["geom_type"]
    geom_params = {
        "R": 1.0,  # Main domain width
        "lc": 0.1,  # Characteristic mesh size
        "tdim": 2,  # Geometric dimension
        "hole_radius": 0.2,
        "h_max": 3.0,
        "h_min": 1.0,
    }
    geom_params["lc"] = parameters["model"].get("ell") / parameters["geometry"].get(
        "mesh_size_factor", 1.0
    )

    with dolfinx.common.Timer(f"~Meshing") as timer:
        gmsh_model, tdim = mesh_circle_with_holes_gmshapi(
            "discwithholes",
            geom_params["R"],
            geom_params["lc"],
            tdim=geom_params["tdim"],
            num_holes=1,
            hole_radius=[geom_params["hole_radius"]],
            hole_positions=[(0.0, 0.0)],
            refinement_factor=0.3,
            order=1,
            # msh_file=msh_file,
            comm=MPI.COMM_WORLD,
        )

        mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    dx = Measure("dx", domain=mesh, subdomain_data=mts)
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
    thickness = dolfinx.fem.Function(V_alpha, name="Thickness")
    t = dolfinx.fem.Constant(mesh, np.array(0.0, dtype=PETSc.ScalarType))
    inner_disp = dolfinx.fem.Function(V_u)
    # inner_disp.interpolate(
    #     lambda x: rotation_displacement(x, (0.0, 0.0), float(t.value))
    # )
    inner_disp.interpolate(lambda x: (x[1], -x[0]))
    # Interpolate the thickness function into the scalar field

    thickness.interpolate(
        lambda x: linear_thickness(
            x,
            geom_params["hole_radius"],
            geom_params["R"],
            H_max=geom_params["h_max"],
            H_min=geom_params["h_min"],
        )
    )

    # plot thickness

    # Set the bounds)
    # plot the inner_disp
    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True
    plotter = pyvista.Plotter(shape=(1, 2))

    plot_vector(inner_disp, plotter, subplot=(0, 1))
    plot_scalar(thickness, plotter, subplot=(0, 1), lineproperties={"cmap": "gray_r"})

    plotter.screenshot(f"{outdir}/inner.png")
    # __import__("pdb").set_trace()
    outer_disp = dolfinx.fem.Constant(
        mesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType)
    )

    for f in [u, alpha_lb, alpha_ub, inner_disp]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    # Bcs
    # Locate boundaries
    outer_dofs_u = locate_dofs_geometrical(
        V_u, lambda x: np.isclose(x[0] ** 2 + x[1] ** 2, geom_params["R"] ** 2)
    )
    outer_dofs_alpha = locate_dofs_geometrical(
        V_alpha, lambda x: np.isclose(x[0] ** 2 + x[1] ** 2, geom_params["R"] ** 2)
    )

    inner_dofs_u = locate_dofs_topological(
        V_u,
        mesh.topology.dim - 1,
        fts.find(10),
    )

    inner_dofs_alpha = locate_dofs_topological(
        V_alpha,
        mesh.topology.dim - 1,
        fts.find(10),
    )

    bcs_u = [
        dirichletbc(outer_disp, outer_dofs_u, V_u),
        dirichletbc(inner_disp, inner_dofs_u),
    ]

    bcs_alpha = [
        dirichletbc(np.array(0.0), inner_dofs_alpha, V_alpha),
        dirichletbc(np.array(0.0), outer_dofs_alpha, V_alpha),
    ]

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    # model = models.DeviatoricSplit(parameters["model"])
    model = models.DamageElasticityModel(parameters["model"])
    dx = Measure("dx", domain=mesh, subdomain_data=mts)
    total_energy = thickness * model.total_energy_density(state) * dx

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

    equilibrium = HybridSolver(
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

    linesearch = LineSearch(
        total_energy,
        state,
        linesearch_parameters=parameters.get("stability").get("linesearch"),
    )

    iterator = StabilityStepper(loads)
    while True:
        try:
            i_t = next(iterator) - 1
        except StopIteration:
            break

        t.value = loads[i_t]
        # inner_disp.interpolate(
        #     lambda x: rotation_displacement(x, (0.0, 0.0), float(t.value))
        # )

        inner_disp.interpolate(
            lambda x: (float(t.value) * x[1], -float(t.value) * x[0])
        )

        alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
        alpha_lb.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        _logger.critical(f"-- Solving for i_t = {i_t} t = {t.value:3.2f} --")
        with dolfinx.common.Timer(f"~First Order: Equilibrium") as timer:
            equilibrium.solve(alpha_lb)

        _logger.critical(f"Bifurcation for t = {t.value:3.2f} --")
        is_unique = bifurcation.solve(alpha_lb)
        is_elastic = not bifurcation._is_critical(alpha_lb)
        inertia = bifurcation.get_inertia()

        z0 = (
            bifurcation._spectrum[0]["xk"]
            if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
            else None
        )

        stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)
        fracture_energy, elastic_energy = postprocess(
            parameters,
            nameExp,
            prefix,
            v,
            β,
            state,
            loads[i_t],
            dx,
            loads,
            equilibrium,
            bifurcation,
            stability,
            inertia=inertia,
            i_t=i_t,
            model=model,
            history_data=history_data,
            signature=signature,
        )

    return None, None, None


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
    parameters["model"]["ell"] = 0.1
    parameters["model"]["w1"] = 3.0

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
        ColorPrint.print_bold(f"===================- {_storage} -=================")
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
