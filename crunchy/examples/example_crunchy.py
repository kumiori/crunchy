from crunchy.experiments import run_experiment
from crunchy.utils import show_image
from crunchy.filters import motion_blur
from crunchy.core import generate_gaussian_field

shape = (200, 200)
blur_distance = 30
field = run_experiment(shape=(200, 200), blur_distance=50)
field = generate_gaussian_field(shape)
blurred_field = motion_blur(field, blur_distance)
# show_image(field)

import numpy as np
from crunchy.filters import (
    motion_blur,
    radial_blur_spin,
    radial_blur_zoom,
    twirl_effect,
    twirl_effect_quadratic,
)
from crunchy.utils import save_image, show_image
from crunchy.core import generate_gaussian_field

# Parameters
shape = (1000, 1000)
blur_distance = 30
spin_amount = 1
zoom_amount = 30
twirl_angle = -60

# Generate Gaussian field
field = generate_gaussian_field(shape)

# Apply Motion Blur
motion_blurred = motion_blur(field, blur_distance)
save_image(motion_blurred, "motion_blur_distance_30.png")
# show_image(motion_blurred)

# Apply Radial Blur (Spin)
spin_blurred = radial_blur_spin(field, amount=spin_amount)
save_image(spin_blurred, f"radial_blur_spin_amount_{spin_amount}.png")
# show_image(spin_blurred)

# Apply Radial Blur (Zoom)
zoom_blurred = radial_blur_zoom(field, amount=zoom_amount)
save_image(zoom_blurred, "radial_blur_zoom_amount_30.png")
# show_image(zoom_blurred)

# Apply Twirl Effect
twirled_field = twirl_effect(field, angle=twirl_angle)
save_image(twirled_field, "twirl_effect_angle_minus_60.png")
# show_image(twirled_field)

# Apply Twirl Effect
twirled_field = twirl_effect_quadratic(field, angle=twirl_angle)
save_image(twirled_field, "twirl_effect_quadratic_angle_minus_60.png")
# show_image(twirled_field)
