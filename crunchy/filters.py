import cv2
import numpy as np


def motion_blur(image, distance=30):
    """Apply motion blur."""
    kernel = np.zeros((distance, distance))
    np.fill_diagonal(kernel, 1)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def radial_blur_spin(image, amount=30):
    h, w = image.shape
    center = (w // 2, h // 2)
    result = image.copy()
    for i in range(amount):
        angle = i / amount * 360
        M = cv2.getRotationMatrix2D(center, angle, 1)
        warped = cv2.warpAffine(result, M, (w, h))
        result = cv2.addWeighted(result, 0.9, warped, 0.1, 0)
    return result


def twirl_effect(image, angle=-60):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    result = np.zeros_like(image)
    y, x = np.meshgrid(np.arange(h), np.arange(w))
    x = x - center[0]
    y = y - center[1]
    theta = np.arctan2(y, x) + angle * np.pi / 180 * (
        np.sqrt(x**2 + y**2) / np.hypot(w, h)
    )
    radius = np.hypot(x, y)
    new_x = (radius * np.cos(theta) + center[0]).astype(np.float32)
    new_y = (radius * np.sin(theta) + center[1]).astype(np.float32)
    result = cv2.remap(
        image,
        new_x,
        new_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return result


def twirl_effect_quadratic(image, angle=-60):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    max_radius = np.hypot(w, h)

    # Create mesh grid
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = x - center[0]
    y = y - center[1]

    # Compute normalized radius
    radius = np.sqrt(x**2 + y**2)
    r_norm = radius / max_radius

    # Quadratic function for twist
    f = np.where(r_norm <= 1, -9 * (r_norm - 2 / 3) ** 2 + 1, 0)

    # Compute new angle
    theta = np.arctan2(y, x) + (angle * np.pi / 180) * f

    # Map back to Cartesian coordinates
    new_x = radius * np.cos(theta) + center[0]
    new_y = radius * np.sin(theta) + center[1]

    # Interpolate using cv2.remap
    new_x = new_x.astype(np.float32)
    new_y = new_y.astype(np.float32)
    result = cv2.remap(
        image,
        new_x,
        new_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return result


def radial_blur_zoom(image, amount=30):
    h, w = image.shape
    center = (w // 2, h // 2)
    result = image.copy()
    for i in range(1, amount + 1):
        scale = 1 + i / (amount * 10)
        M = cv2.getRotationMatrix2D(center, 0, scale)
        warped = cv2.warpAffine(result, M, (w, h))
        result = cv2.addWeighted(result, 0.9, warped, 0.1, 0)
    return result
