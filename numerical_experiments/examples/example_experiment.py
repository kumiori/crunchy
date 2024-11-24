from numerical_experiments.experiments import run_experiment
from numerical_experiments.utils import show_image

field = run_experiment(shape=(200, 200), blur_distance=50)
show_image(field)
