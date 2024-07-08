import json
from cpu_vector_addition import cpu_vector_addition_single_thread, cpu_vector_addition_multithread_numba
from gpu_vector_addition import gpu_vector_addition_cupy, gpu_vector_addition_cupy_custom_kernel, \
    gpu_vector_addition_numba


def compare_performance(n_values, num_runs=10):
    results = {
        "dimensions": n_values,
        "cpu_single_thread": [],
        "cpu_multi_thread_vanilla": [],
        "cpu_multi_thread_numba": [],
        "gpu_cupy": [],
        "gpu_custom_kernel": [],
        "gpu_numba": []
    }

    for n in n_values:
        cpu_times_single = []
        cpu_times_multi_numba = []
        #cpu_times_multi_vanilla = []
        gpu_times_cupy = []
        gpu_times_custom_kernel = []
        gpu_times_numba = []

        for _ in range(num_runs):
            cpu_times_single.append(cpu_vector_addition_single_thread(n))
            #cpu_times_multi_vanilla.append(cpu_vector_addition_vanilla_python(n))
            cpu_times_multi_numba.append(cpu_vector_addition_multithread_numba(n))
            gpu_times_cupy.append(gpu_vector_addition_cupy(n))
            gpu_times_custom_kernel.append(gpu_vector_addition_cupy_custom_kernel(n))
            gpu_times_numba.append(gpu_vector_addition_numba(n))

        results["cpu_single_thread"].append(sum(cpu_times_single) / num_runs)
        #results["cpu_multi_thread_vanilla"].append(sum(cpu_times_multi_vanilla) / num_runs)
        results["cpu_multi_thread_numba"].append(sum(cpu_times_multi_numba) / num_runs)
        results["gpu_cupy"].append(sum(gpu_times_cupy) / num_runs)
        results["gpu_custom_kernel"].append(sum(gpu_times_custom_kernel) / num_runs)
        results["gpu_numba"].append(sum(gpu_times_numba) / num_runs)

        print(f"Dimension: {n}")
        print(f"CPU (single-threaded) average: {results['cpu_single_thread'][-1]:.10f} seconds")
        print(f"CPU (multi-threaded-numba) average: {results['cpu_multi_thread_numba'][-1]:.10f} seconds")
        #print(f"CPU (multi-threaded-vanilla) average: {results['cpu_multi_thread_vanilla'][-1]:.6f} seconds")
        print(f"GPU (CuPy) average: {results['gpu_cupy'][-1]:.10f} seconds")
        print(f"GPU (Custom Kernel) average: {results['gpu_custom_kernel'][-1]:.10f} seconds")
        print(f"GPU (Numba) average: {results['gpu_numba'][-1]:.10f} seconds")

    with open("performance_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    n_values = [10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8]  # Different dimensions
    compare_performance(n_values, 10)
