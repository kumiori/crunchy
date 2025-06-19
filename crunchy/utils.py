import matplotlib.pyplot as plt
import os
from pathlib import Path
from mpi4py import MPI
import yaml

import dolfinx
import basix
from dolfinx.io import XDMFFile
from dolfinx.fem import Function
import ufl
import petsc4py.PETSc as PETSc
import json
from typing import List
import numpy as np
import pandas as pd


def save_image(image, filename):
    """Save an image to a file."""
    plt.imsave(filename, image, cmap="gray")


def show_image(image):
    """Display an image using matplotlib."""
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def setup_output_directory(storage, parameters, outdir):
    __import__("pdb").set_trace()
    if storage is None:
        prefix = os.path.join(
            outdir, f"1d-{parameters['geometry']['geom_type']}-first-new-hybrid"
        )
    else:
        prefix = storage

    if MPI.COMM_WORLD.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    return prefix


def save_params_to_yaml(params, filename):
    """
    Save the updated params dictionary to a YAML file.

    Args:
    - params (dict): Dictionary containing all parameters.
    - filename (str): Path to the YAML file to save.
    """
    with open(filename, "w") as file:
        yaml.dump(params, file, default_flow_style=False)


def update_parameters(parameters, key, value):
    """
    Recursively traverses the dictionary d to find and update the key's value.

    Args:
    d (dict): The dictionary to traverse.
    key (str): The key to find and update.
    value: The new value to set for the key.

    Returns:
    bool: True if the key was found and updated, False otherwise.
    """
    if key in parameters:
        parameters[key] = value
        return True

    for k, v in parameters.items():
        if isinstance(v, dict):
            if update_parameters(v, key, value):
                return True

    return False


def table_timing_data(tasks=None):
    import pandas as pd
    from dolfinx.common import timing

    timing_data = []
    if tasks is None:
        tasks = [
            "~Mesh Generation",
            "~First Order: min-max equilibrium",
            "~Postprocessing and Vis",
            "~Computation Experiment",
        ]

    for task in tasks:
        timing_data.append(timing(task))

    df = pd.DataFrame(
        timing_data, columns=["reps", "wall tot", "usr", "sys"], index=tasks
    )

    return df


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


def new_history_data():
    return {
        "load": [],
        "elastic_energy": [],
        "fracture_energy": [],
        "total_energy": [],
        "equilibrium_data": [],
        "cone_data": [],
        "eigs_ball": [],
        "eigs_cone": [],
        "stable": [],
        "unique": [],
        "inertia": [],
        "y_norms": [],
    }


def write_history_data(
    equilibrium,
    bifurcation,
    stability,
    history_data,
    t,
    inertia,
    stable,
    energies: List,
):
    elastic_energy = energies[0]
    fracture_energy = energies[1]
    unique = True if inertia[0] == 0 and inertia[1] == 0 else False

    history_data["load"].append(t)
    history_data["fracture_energy"].append(fracture_energy)
    history_data["elastic_energy"].append(elastic_energy)
    history_data["total_energy"].append(elastic_energy + fracture_energy)
    history_data["equilibrium_data"].append(equilibrium.data)
    history_data["inertia"].append(inertia)
    history_data["unique"].append(unique)
    history_data["stable"].append(stable)
    history_data["eigs_ball"].append(
        bifurcation.data["eigs"] if bifurcation.data else None
    )
    history_data["cone_data"].append(stability.data)
    history_data["eigs_cone"].append(stability.solution["lambda_t"])

    return


def dump_output(
    _nameExp,
    prefix,
    history_data,
    u,
    alpha,
    equilibrium,
    bifurcation,
    stability,
    inertia,
    t,
    perturbation_log,
    fracture_energy,
    elastic_energy,
):
    comm = MPI.COMM_WORLD
    with dolfinx.common.Timer(f"~Output and Storage") as timer:
        with XDMFFile(
            comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        ) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()

        experimental_data = pd.DataFrame(history_data)

        if comm.rank == 0:
            experimental_data.to_pickle(f"{prefix}/experimental_data.pkl")

        with open(f"{prefix}/perturbation_log.json", "w") as f:
            json.dump(perturbation_log, f, indent=4)


import numpy as np
from petsc4py import PETSc

# TODO: implement nullspace


def build_rigid_body_modes(V_u):
    """
    Construct the null space for a 2D problem in Dolfinx.
    - Two translations (x, y)
    - One rotation about the z-axis
    """
    dof_coords = V_u.tabulate_dof_coordinates()
    num_dofs = V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs

    # Initialize nullspace matrix
    nullspace = np.zeros((num_dofs, 3), dtype=np.float64)

    # Translation in X (index 0 in 2D)
    nullspace[:, 0] = 1.0  # Uniform motion in x-direction

    # Translation in Y (index 1 in 2D)
    nullspace[:, 1] = 1.0  # Uniform motion in y-direction

    # Rotation about Z-axis: displacement u_x = -y, u_y = x
    nullspace[:, 2] = -dof_coords[:, 1]  # -y for x-component
    nullspace[:, 2] += dof_coords[:, 0]  # +x for y-component

    return nullspace


def set_nullspace(solver, V_u):
    """
    Sets the rigid body nullspace (2 translations + 1 rotation) to the SNES solver.
    """
    # Get rigid body modes
    nullspace_matrix = build_rigid_body_modes(V_u)

    # Convert to PETSc Vecs
    nullspace_vectors = []
    for i in range(nullspace_matrix.shape[1]):
        vec = PETSc.Vec().createMPI(nullspace_matrix.shape[0])
        vec.setArray(nullspace_matrix[:, i])
        vec.assemble()
        nullspace_vectors.append(vec)

    # Create the PETSc NullSpace object
    null_space = PETSc.NullSpace().create(vectors=nullspace_vectors)

    # Assign nullspace to the solver's matrix (A = solver.jacobian)
    solver.A.setNullSpace(null_space)
    solver.A.setTransposeNullSpace(null_space)


# # Create SNES solver
# snes_solver = SNESSolver(energy_u, u, bcs, ...)
# set_nullspace(snes_solver, V_u)


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
