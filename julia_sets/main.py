import json
import numpy as np
import matplotlib.pyplot as plt

from cpu_julia import julia_set_single_thread, julia_set_multithread_with_timing
from gpu_julia import julia_set_numba_with_timing, julia_set_cupy, julia_set_cupy_custom_kernel


def plot_julia(img, width, height):
    """
    Plot the Julia set image.

    Parameters:
        img (np.ndarray): The computed Julia set image.
        width (int): The width of the image.
        height (int): The height of the image.
    """
    dpi = 250
    fig_size = (width / dpi, height / dpi)
    plt.figure(figsize=fig_size, dpi=dpi)
    plt.imshow(img, cmap='hot', extent=(0, 1, 0, 1), interpolation='nearest')
    plt.colorbar()
    plt.title("Julia Set")
    plt.savefig("julia_classic_quadratic.png", dpi=dpi)
    plt.show()


def save_performance_results(results):
    with open("performance_results.json", "w") as f:
        json.dump(results, f)


def generate_julia_set_image(width, height, x_min, x_max, y_min, y_max, c, max_iter):
    # Best performing function: julia_set_cupy_custom_kernel
    img, time_taken = julia_set_cupy_custom_kernel(width, height, x_min, x_max, y_min, y_max, c, max_iter)
    print("Time taken for computing Julia: ", time_taken)
    plot_julia(img, width, height)


def run():
    # Parameters for the Julia set
    resolutions = [(720, 480), (1280, 720), (1920, 1080), (2048, 1080), (3840, 2160)]  # SD, HD, FULL HD, 2K, 4K
    iteration_counts = [100, 256, 512]
    num_runs = 10

    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    c = complex(-0.7, 0.27015)

    # Initialize performance results storage
    results = {
        "resolutions": resolutions,
        "iteration_counts": iteration_counts,
        "cpu_single_thread": [],
        "cpu_multi_thread_numba": [],
        "gpu_cupy": [],
        "gpu_numba": [],
        "gpu_custom_kernel": []
    }

    for width, height in resolutions:
        for max_iter in iteration_counts:
            cpu_times_single = []
            cpu_times_multi_thread_numba = []
            gpu_times_numba = []
            gpu_times_cupy = []
            gpu_times_custom_kernel = []

            for _ in range(num_runs):
                # Single-threaded CPU
                _, time_taken = julia_set_single_thread(width, height, x_min, x_max, y_min, y_max, c, max_iter)
                cpu_times_single.append(time_taken)

                # Multi-threaded CPU (Numba)
                _, time_taken = julia_set_multithread_with_timing(width, height, x_min, x_max, y_min, y_max, c,
                                                                  max_iter)
                cpu_times_multi_thread_numba.append(time_taken)

                # GPU (Numba)
                _, time_taken = julia_set_numba_with_timing(width, height, x_min, x_max, y_min, y_max, c, max_iter)
                gpu_times_numba.append(time_taken)

                # GPU (CuPy)
                _, time_taken = julia_set_cupy(width, height, x_min, x_max, y_min, y_max, c, max_iter)
                gpu_times_cupy.append(time_taken)

                # GPU (Custom Kernel)
                _, time_taken = julia_set_cupy_custom_kernel(width, height, x_min, x_max, y_min, y_max, c, max_iter)
                gpu_times_custom_kernel.append(time_taken)

            # Calculate average times
            avg_cpu_single = sum(cpu_times_single) / num_runs
            avg_cpu_multi_thread_numba = sum(cpu_times_multi_thread_numba) / num_runs
            avg_gpu_numba = sum(gpu_times_numba) / num_runs
            avg_gpu_cupy = sum(gpu_times_cupy) / num_runs
            avg_gpu_custom_kernel = sum(gpu_times_custom_kernel) / num_runs

            results["cpu_single_thread"].append(
                {"resolution": (width, height), "iterations": max_iter, "time": avg_cpu_single})
            results["cpu_multi_thread_numba"].append(
                {"resolution": (width, height), "iterations": max_iter, "time": avg_cpu_multi_thread_numba})
            results["gpu_numba"].append({"resolution": (width, height), "iterations": max_iter, "time": avg_gpu_numba})
            results["gpu_cupy"].append({"resolution": (width, height), "iterations": max_iter, "time": avg_gpu_cupy})
            results["gpu_custom_kernel"].append(
                {"resolution": (width, height), "iterations": max_iter, "time": avg_gpu_custom_kernel})

            print(
                f"Single-threaded CPU average time ({width}x{height}, {max_iter} iterations): {avg_cpu_single:.6f} seconds")
            print(
                f"Multithread-threaded CPU (Numba) average time ({width}x{height}, {max_iter} iterations): {avg_cpu_multi_thread_numba:.6f} seconds")
            print(
                f"Multithread-threaded GPU (NUMBA) average time ({width}x{height}, {max_iter} iterations): {avg_gpu_numba:.6f} seconds")
            print(
                f"Multithread-threaded GPU (CuPy) average time ({width}x{height}, {max_iter} iterations): {avg_gpu_cupy:.6f} seconds")
            print(
                f"Multithread-threaded GPU (Custom Kernel) average time ({width}x{height}, {max_iter} iterations): {avg_gpu_custom_kernel:.6f} seconds")

    # Save the results
    save_performance_results(results)


if __name__ == "__main__":
    is_continuing = True
    while is_continuing:
        print('1. Generate a custom Julia set;\n2. Run the tests;\n3. Exist;')
        option = input('Choose an option: ').lower()

        match option:
            case "1" | "generate":
                width = int(input('width: '))
                height = int(input('height: '))
                max_iter = int(input('max_iter: '))
                x_min, x_max = float(input('x_min: ')), float(input('x_max: '))
                y_min, y_max = float(input('y_min: ')), float(input('y_max: '))
                c = input("Complex number (r+j; ex: -0.7+0.27015j): ")
                c = complex(c)

                generate_julia_set_image(width=width, height=height, max_iter=max_iter, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, c=c)
            case "2" | "run":
                run()
            case "3" | "exit":
                is_continuing = False
            case _:
                print("Unknown command")
