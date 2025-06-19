import numpy as np

import numpy as np
from dolfinx.fem import Function
from petsc4py import PETSc


def generate_gaussian_function(V, mean=0.0, std=1.0, seed=None):
    """
    Generate a random Gaussian noise field in parallel and interpolate it onto a function space.

    Parameters:
        V: dolfinx.fem.FunctionSpace
            Function space to interpolate the noise onto.
        mean: float
            Mean of the Gaussian distribution.
        std: float
            Standard deviation of the Gaussian distribution.
        seed: int, optional
            Random seed for reproducibility.

    Returns:
        Function
            A Dolfinx Function containing the parallel-consistent random Gaussian noise.
    """
    # Get the local vector size
    local_size = V.dofmap.index_map.size_local
    global_size = V.dofmap.index_map.size_global
    rank = V.mesh.comm.rank

    # Set random seed for reproducibility across ranks
    if seed is not None:
        global_seed = seed + rank  # Ensure unique seeds per rank
    else:
        global_seed = None

    rng = np.random.default_rng(global_seed)

    # Generate local Gaussian noise
    local_noise = rng.normal(loc=mean, scale=std, size=local_size)

    # Create a PETSc vector for global noise
    noise = Function(V, name="Noise")
    global_array = noise.x.petsc_vec.array

    # Distribute the local noise to the global vector
    global_array[:] = 0.0  # Initialize
    global_array[:local_size] = local_noise

    # Synchronize ghost values
    noise.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    return noise


def generate_gaussian_field(shape=(100, 100), mean=0, std=1):
    """Generate a 2D Gaussian field."""
    return np.random.normal(mean, std, size=shape)


import basix
import dolfinx


def create_function_spaces_2d(mesh):
    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))
    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)

    V_u = dolfinx.fem.functionspace(mesh, element_u)
    V_alpha = dolfinx.fem.functionspace(mesh, element_alpha)

    return V_u, V_alpha


def create_function_spaces_nd(mesh, dim):
    """
    Create function spaces for displacement and scalar fields in an arbitrary dimension.

    Parameters:
    - mesh: The mesh object for the domain.
    - dim: The spatial dimension of the problem (e.g., 1, 2, or 3).

    Returns:
    - V_u: Function space for vector fields (displacement).
    - V_alpha: Function space for scalar fields (e.g., damage or pressure).
    """
    # Define the vector element for displacement (dim components)
    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(dim,))
    # Define the scalar element for damage or other scalar fields
    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)

    # Create function spaces
    V_u = dolfinx.fem.functionspace(mesh, element_u)
    V_alpha = dolfinx.fem.functionspace(mesh, element_alpha)

    return V_u, V_alpha


def initialise_functions(V_u, V_alpha):
    u = dolfinx.fem.Function(V_u, name="Displacement")
    u_ = dolfinx.fem.Function(V_u, name="BoundaryDisplacement")
    alpha = dolfinx.fem.Function(V_alpha, name="Damage")
    beta = dolfinx.fem.Function(V_alpha, name="DamagePerturbation")
    v = dolfinx.fem.Function(V_u, name="DisplacementPerturbation")
    state = {"u": u, "alpha": alpha}

    return u, u_, alpha, beta, v, state


import hashlib
import yaml


def save_parameters(parameters, prefix):
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    if MPI.COMM_WORLD.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)


import os
from pathlib import Path
from mpi4py import MPI


def setup_output_directory(storage, parameters, outdir):
    if outdir is None:
        outdir = "output"
    if storage is None:
        prefix = os.path.join(
            outdir,
            f"{parameters['geometry']['geometric_dimension']}d-{parameters['geometry']['geom_type']}",
        )
    else:
        prefix = storage

    if MPI.COMM_WORLD.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    return prefix
