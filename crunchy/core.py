import numpy as np


def generate_gaussian_field(shape=(100, 100), mean=0, std=1):
    """Generate a 2D Gaussian field."""
    return np.random.normal(mean, std, size=shape)


import basix
import dolfinx


def create_function_spaces_2d(mesh):
    # element_u = VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=2)
    # element_alpha = FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))
    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)

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
