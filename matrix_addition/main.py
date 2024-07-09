import json
from cpu_matrix_addition import cpu_matrix_addition_single_thread, cpu_matrix_addition_multithread_numba
from gpu_matrix_addition import gpu_matrix_addition_cupy, gpu_matrix_addition_cupy_custom_kernel, gpu_matrix_addition_numba

def compare_performance(n_values, num_runs=10):
    results = {
        "dimensions": n_values,
        "cpu_single_thread": [],
        "cpu_multi_thread_numba": [],
        "gpu_cupy": [],
        "gpu_custom_kernel": [],
        "gpu_numba": []
    }

    for n in n_values:
        cpu_times_single = []
        cpu_times_multi_numba = []
        gpu_times_cupy = []
        gpu_times_custom_kernel = []
        gpu_times_numba = []

        for _ in range(num_runs):
            t_single = cpu_matrix_addition_single_thread(n)[1]
            t_multi_numba = cpu_matrix_addition_multithread_numba(n)[1]
            t_cupy = gpu_matrix_addition_cupy(n)
            t_custom_kernel = gpu_matrix_addition_cupy_custom_kernel(n)
            t_numba = gpu_matrix_addition_numba(n)

            cpu_times_single.append(t_single)
            cpu_times_multi_numba.append(t_multi_numba)
            gpu_times_cupy.append(t_cupy)
            gpu_times_custom_kernel.append(t_custom_kernel)
            gpu_times_numba.append(t_numba)

        results["cpu_single_thread"].append(sum(cpu_times_single) / num_runs)
        results["cpu_multi_thread_numba"].append(sum(cpu_times_multi_numba) / num_runs)
        results["gpu_cupy"].append(sum(gpu_times_cupy) / num_runs)
        results["gpu_custom_kernel"].append(sum(gpu_times_custom_kernel) / num_runs)
        results["gpu_numba"].append(sum(gpu_times_numba) / num_runs)

        print(f"Dimension: {n}x{n}")
        print(f"CPU (single-threaded) average: {results['cpu_single_thread'][-1]:.10f} seconds")
        print(f"CPU (multi-threaded-numba) average: {results['cpu_multi_thread_numba'][-1]:.10f} seconds")
        print(f"GPU (CuPy) average: {results['gpu_cupy'][-1]:.10f} seconds")
        print(f"GPU (Custom Kernel) average: {results['gpu_custom_kernel'][-1]:.10f} seconds")
        print(f"GPU (Numba) average: {results['gpu_numba'][-1]:.10f} seconds")

    with open("performance_results_matrix.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    n_values = [50, 100, 200, 400, 800, 1600, 3200, 6400]  # Different matrix dimensions
    compare_performance(n_values, 10)
