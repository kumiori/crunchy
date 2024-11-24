from numerical_experiments.core import generate_gaussian_field


def test_generate_gaussian_field():
    field = generate_gaussian_field((10, 10), mean=5, std=2)
    assert field.shape == (10, 10)
    assert abs(field.mean() - 5) < 1
