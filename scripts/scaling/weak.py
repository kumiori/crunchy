import os
import subprocess
import numpy as np


# Weak scaling setup
def run_weak_scaling(base_problem_size, num_processors_list):
    """
    Perform weak scaling tests, aiming to maintain a constant workload per processor, increasing both problem size and the number of processors proportionally.

    Parameters:
    - base_problem_size: int, the problem size for a single processor.
    - num_processors_list: list of ints, the number of processors to test.
    """
    for num_procs in num_processors_list:
        # Scale problem size linearly with number of processors
        problem_size = base_problem_size * num_procs

        # Command to execute the elasticity test
        command = [
            "mpirun",
            "-np",
            str(num_procs),
            "python",
            "elasticity.py",
            "--problem_size",
            str(problem_size),
            "--num_procs",
            str(num_procs),
        ]

        print(
            f"Running weak scaling test with {num_procs} processors and problem size {problem_size}"
        )
        subprocess.run(command)


if __name__ == "__main__":
    base_problem_size = 10000  # Adjust as needed
    num_processors_list = [1, 2, 4, 8, 16]  # Example processor counts
    run_weak_scaling(base_problem_size, num_processors_list)
