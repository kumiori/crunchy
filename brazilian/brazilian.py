import os
import logging
from mpi4py import MPI
import dolfinx
from irrevolutions.utils import (
    Visualization,
    ColorPrint,
)
from dolfinx.common import list_timings

# from irrevolutions.models import default_model_parameters
from dolfinx.io import XDMFFile, gmshio
import irrevolutions.models as models
from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.algorithms.ls import StabilityStepper, LineSearch
from irrevolutions.meshes.boolean import create_disk_with_hole
from irrevolutions.meshes.primitives import mesh_bar_gmshapi
from dolfinx.mesh import CellType
from crunchy.core import (
    initialise_functions,
    create_function_spaces_nd,
    setup_output_directory,
    save_parameters,
    initialise_functions,
)
from dolfinx.fem import Function, locate_dofs_topological, dirichletbc
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
    t = dolfinx.fem.Constant(mesh, np.array(0.0, dtype=PETSc.ScalarType))
    top_disp = dolfinx.fem.Constant(mesh, np.array([0.0, t], dtype=PETSc.ScalarType))
    bottom_disp = dolfinx.fem.Constant(
        mesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType)
    )
    # Define the state
    u_zero = Function(V_u, name="BoundaryDatum")
    u_zero.interpolate(lambda x: [np.zeros_like(x[0]), np.zeros_like(x[1])])

    for f in [u, u_zero, alpha_lb, alpha_ub]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
    # Bcs

    # Locate top and bottom boundaries
    def top_boundary(x):
        return np.isclose(x[1], geom_params["R_outer"], atol=1e-3)

    def bottom_boundary(x):
        return np.isclose(x[1], -geom_params["R_outer"], atol=1e-3)

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
        dirichletbc(top_disp, top_dofs, V_u),
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
    # iterator = StabilityStepper(loads)
    # while True:
    #     try:
    #         i_t, t = next(iterator)
    #     except StopIteration:
    #         break

    #     eps_t.value = t
    #     u_zero.interpolate(lambda x: radial_field(x) * eps_t)
    #     u_zero.vector.ghostUpdate(
    #         addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    #     )

    #     alpha.vector.copy(alpha_lb.vector)
    #     alpha_lb.vector.ghostUpdate(
    #         addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    #     )

    #     _logger.critical(f"-- Solving for t = {t:3.2f} --")
    #     with dolfinx.common.Timer(f"~First Order: Equilibrium") as timer:
    #         equilibrium.solve(alpha_lb)

    #     _logger.critical(f"Bifurcation for t = {t:3.2f} --")
    #     is_unique = bifurcation.solve(alpha_lb)
    #     is_elastic = not bifurcation._is_critical(alpha_lb)
    #     inertia = bifurcation.get_inertia()

    #     z0 = (
    #         bifurcation._spectrum[0]["xk"]
    #         if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
    #         else None
    #     )
    #     ColorPrint.print_bold(f"Evolution is unique: {is_unique}")
    #     ColorPrint.print_bold(f"State's inertia: {inertia}")
    #     # stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)
    #     equilibrium.log()
    #     bifurcation.log()
    #     ColorPrint.print_bold(f"===================- {_storage} -=================")

    #     stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)

    #     equilibrium.log()
    #     bifurcation.log()
    #     stability.log()

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
    parameters["loading"]["max"] = 1.0
    parameters["loading"]["steps"] = 10

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
