import os
import subprocess


# Strong scaling setup
def run_strong_scaling(problem_size, num_processors_list):
    """
    Perform strong scaling tests, keeping the problem size fixed while varying the number of processors.

    Parameters:
    - problem_size: int, the fixed problem size for all runs.
    - num_processors_list: list of ints, the number of processors to test.
    """
    for num_procs in num_processors_list:
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
            f"Running strong scaling test with {num_procs} processors and problem size {problem_size}"
        )
        subprocess.run(command)


if __name__ == "__main__":
    problem_size = 100000  # Adjust as needed
    num_processors_list = [1, 2, 4, 8, 16]  # Example processor counts
    run_strong_scaling(problem_size, num_processors_list)
