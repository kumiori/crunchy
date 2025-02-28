import os
import logging
from mpi4py import MPI
import dolfinx
from irrevolutions.utils import (
    Visualization,
    ColorPrint,
)
from dolfinx.common import list_timings
from matplotlib import cm

# from irrevolutions.models import default_model_parameters
from dolfinx.io import XDMFFile, gmshio
import irrevolutions.models as models
from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.algorithms.ls import StabilityStepper, LineSearch
from irrevolutions.meshes.boolean import create_disk_with_hole
from irrevolutions.utils.plots import (
    plot_AMit_load,
    plot_energies,
    plot_force_displacement,
)
from crunchy.plots import plot_spectrum
from irrevolutions.utils.viz import plot_scalar, plot_vector
import matplotlib.pyplot as plt
import pyvista
import ufl
from dolfinx.plot import vtk_mesh as compute_topology

# from irrevolutions.meshes.primitives import mesh_bar_gmshapi
from irrevolutions.utils import _logger, setup_logger_mpi, history_data
from dolfinx.mesh import CellType, locate_entities, meshtags
from crunchy.core import (
    initialise_functions,
    create_function_spaces_nd,
    setup_output_directory,
    save_parameters,
    initialise_functions,
)
from crunchy.utils import dump_output, write_history_data, new_history_data

from dolfinx.fem import (
    Function,
    locate_dofs_topological,
    dirichletbc,
    assemble_scalar,
    form,
)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    Measure,
)
import numpy as np
import yaml
import hashlib

comm = MPI.COMM_WORLD
model_rank = 0


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


def positive_negative_trace(eps):
    """
    Compute the positive and negative parts of the trace of the strain tensor.
    """
    tr_eps = ufl.tr(eps)
    tr_plus = ufl.max_value(tr_eps, 0)
    tr_minus = ufl.min_value(tr_eps, 0)
    return tr_plus, tr_minus


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

            plotter.screenshot(f"{_storage}/stess_state_t{i_t:03}.png")

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
            _plt, grid = plot_vector(u, plotter, subplot=(0, 1))
            _plt.screenshot(f"{_storage}/state_t{i_t:03}.png")

            try:
                fig, ax = plot_spectrum(history_data)
                fig.savefig(f"{_storage}/spectrum.png")
                plt.close(fig)
            except Exception as e:
                _logger.error(f"Error plotting spectrum: {e}")
    return fracture_energy, elastic_energy


def run_computation(parameters, storage):
    nameExp = parameters["geometry"]["geom_type"]
    geom_params = {
        "R_outer": 1.0,  # Outer disk radius
        "R_inner": 0.3,  # Inner hole radius (set to 0.0 for no hole)
        "lc": 0.05,  # Mesh element size
        "a": 0.1,  # Half-width of the refined region (-a < x < a)
    }
    with dolfinx.common.Timer(f"~Meshing") as timer:
        gmsh_model, tdim = create_disk_with_hole(comm, geom_params)
        # gmsh_model, tdim = mesh_bar_gmshapi("bar", 1, 0.1, 0.1, 2)

        mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)
    # mesh = dolfinx.mesh.create_rectangle(
    #     MPI.COMM_WORLD, [[0.0, 0.0], [1, 0.1]], [100, 10], cell_type=CellType.triangle
    # )
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
        return np.isclose(x[1], geom_params["R_outer"], atol=1e-3)

    def bottom_boundary(x):
        return np.isclose(x[1], -geom_params["R_outer"], atol=1e-3)

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

    model = models.DeviatoricSplit(parameters["model"])
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
        # top_disp = dolfinx.fem.Constant(mesh, np.array([0.0, -t.value]))
        top_disp.interpolate(
            lambda x: np.stack(
                [np.zeros_like(x[0]), np.full_like(x[0], -t.value)], axis=0
            )
        )
        #     u_zero.interpolate(lambda x: radial_field(x) * eps_t)
        #     u_zero.vector.ghostUpdate(
        #         addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        #     )
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

        # ColorPrint.print_bold(f"Evolution is unique: {is_unique}")
        # ColorPrint.print_bold(f"State's inertia: {inertia}")
        stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)
        # equilibrium.log()
        bifurcation.log()
        stability.log()
        ColorPrint.print_bold(f"===================- {_storage} -=================")

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


def load_parameters(file_path):
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

    parameters["model"]["model_dimension"] = 2
    parameters["model"]["model_type"] = "2D"

    # parameters["model"]["at_number"] = 1
    parameters["loading"]["min"] = 0.0
    parameters["loading"]["max"] = 0.1
    parameters["loading"]["steps"] = 3

    parameters["geometry"]["geom_type"] = "circle"
    parameters["geometry"]["mesh_size_factor"] = 3
    parameters["geometry"]["R_outer"] = 1.0  # Outer disk radius
    parameters["geometry"]["R_inner"] = 0.3  # Inner hole radius (0.0 for no hole)
    parameters["geometry"]["lc"] = 0.05  # Mesh element size
    parameters["geometry"]["a"] = 0.1  # Half-width of the refined region (-a < x < a)

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-3

    parameters["model"]["w1"] = 1
    parameters["model"]["k_res"] = 0.0
    parameters["model"]["mu"] = 1
    parameters["model"]["ell"] = 0.02
    parameters["solvers"]["damage_elasticity"]["max_it"] = 1000

    parameters["solvers"]["damage_elasticity"]["alpha_rtol"] = 1e-3
    parameters["solvers"]["newton"]["snes_atol"] = 1e-8
    parameters["solvers"]["newton"]["snes_rtol"] = 1e-8

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # Load parameters
    parameters, signature = load_parameters("parameters.yaml")

    # Run computation
    _storage = f"./output/MPI-{MPI.COMM_WORLD.Get_size()}/{signature[0:6]}"
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

    _timings = table_timing_data()
    visualization.save_table(_timings, "timing_data")
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    ColorPrint.print_bold(f"===================- {signature} -=================")
    ColorPrint.print_bold(f"===================- {_storage} -=================")
