#!/usr/bin/env python3

from typing import Optional
import dolfinx
import ufl

from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)


from pathlib import Path
import hashlib
import yaml
import os
from mpi4py import MPI


def setup_output_directory(storage, parameters, outdir):
    if storage is None:
        prefix = os.path.join(
            outdir, f"1d-{parameters['geometry']['geom_type']}-first-new-hybrid"
        )
    else:
        prefix = storage

    if MPI.COMM_WORLD.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    return prefix


def save_parameters(parameters, prefix):
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    if MPI.COMM_WORLD.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    return signature


# from ufl import VectorElement, FiniteElement, Measure
from dolfinx import mesh, fem
