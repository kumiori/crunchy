from setuptools import setup, find_packages

setup(
    name="Crunchy",
    version="0.1",
    description="A package for crafting numerical experiments.",
    author="Andrés A. León Baldelli",
    author_email="leon.baldelly@cnrs.fr",
    packages=find_packages(),
    install_requires=["numpy", "opencv-python", "matplotlib", "dolfinx", "mpi4py"],
    python_requires=">=3.8",
)
