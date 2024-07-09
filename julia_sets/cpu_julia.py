import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit, prange


def julia_set_single_thread(width, height, x_min, x_max, y_min, y_max, c, max_iter):
    """
    Compute the Julia set using a single-threaded CPU approach.

    Parameters:
        width (int): The width of the image.
        height (int): The height of the image.
        x_min (float): The minimum x-value of the complex plane.
        x_max (float): The maximum x-value of the complex plane.
        y_min (float): The minimum y-value of the complex plane.
        y_max (float): The maximum y-value of the complex plane.
        c (complex): The complex constant c in the Julia set formula.
        max_iter (int): The maximum number of iterations.

    Returns:
        img (np.ndarray): The computed Julia set image.
        time_taken (float): The time taken to compute the Julia set.
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    img = np.zeros(Z.shape, dtype=int)

    start_time = time.time()
    for i in range(max_iter):
        mask = np.abs(Z) < 4.0
        Z[mask] = Z[mask] ** 2 + c
        img[mask] = i
    end_time = time.time()

    time_taken = end_time - start_time
    return img, time_taken


@njit(parallel=True)
def julia_set_multi_thread(width, height, x_min, x_max, y_min, y_max, c, max_iter):
    """
    Compute the Julia set using a multi-threaded CPU approach with Numba.

    Parameters:
        width (int): The width of the image.
        height (int): The height of the image.
        x_min (float): The minimum x-value of the complex plane.
        x_max (float): The maximum x-value of the complex plane.
        y_min (float): The minimum y-value of the complex plane.
        y_max (float): The maximum y-value of the complex plane.
        c (complex): The complex constant c in the Julia set formula.
        max_iter (int): The maximum number of iterations.

    Returns:
        img (np.ndarray): The computed Julia set image.
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    img = np.zeros((height, width), dtype=np.int32)

    for i in prange(height):
        for j in prange(width):
            zx, zy = x[j], y[i]
            z = zx + 1j * zy
            iteration = 0
            while abs(z) < 4 and iteration < max_iter:
                z = z ** 2 + c
                iteration += 1
            img[i, j] = iteration

    return img

def julia_set_multithread_with_timing(width, height, x_min, x_max, y_min, y_max, c, max_iter):
    """
    Compute the Julia set using a multi-threaded CPU approach with Numba and measure the time taken.

    Parameters:
        width (int): The width of the image.
        height (int): The height of the image.
        x_min (float): The minimum x-value of the complex plane.
        x_max (float): The maximum x-value of the complex plane.
        y_min (float): The minimum y-value of the complex plane.
        y_max (float): The maximum y-value of the complex plane.
        c (complex): The complex constant c in the Julia set formula.
        max_iter (int): The maximum number of iterations.

    Returns:
        img (np.ndarray): The computed Julia set image.
        time_taken (float): The time taken to compute the Julia set.
    """
    start_time = time.time()
    img = julia_set_multi_thread(width, height, x_min, x_max, y_min, y_max, c, max_iter)
    end_time = time.time()
    time_taken = end_time - start_time

    return img, time_taken