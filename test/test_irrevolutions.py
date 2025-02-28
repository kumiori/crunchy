import sys
# from importlib import reload

print(sys.path)
# Remove local paths that cause conflicts
# sys.path = [p for p in sys.path if "mec647/src" not in p]

# Reload the module to ensure the correct version is used
import irrevolutions
import dolfinx
import numba
import numpy

print("NumPy:", numpy.__version__, "Numba:", numba.__version__)

print("Now using irrevolutions from:", irrevolutions.__file__)
# expected output
# Now using irrevolutions from: /opt/anaconda3/envs/dolfinx-0.9/lib/python3.13/site-packages/irrevolutions/__init__.py
