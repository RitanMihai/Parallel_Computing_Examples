import json
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np

class Method(Enum):
    CPU_SINGLE_THREAD = "CPU Single Thread"
    CPU_MULTI_THREAD_NUMBA = "CPU Multi Thread Numba"
    GPU_CUPY = "GPU CuPy"
    GPU_CUSTOM_KERNEL = "GPU Custom Kernel"
    GPU_NUMBA = "GPU Numba"

def plot_performance(methods_to_plot):
    with open("performance_results_matrix.json", "r") as f:
        results = json.load(f)

    dimensions = results["dimensions"]

    method_data = {
        Method.CPU_SINGLE_THREAD: results["cpu_single_thread"],
        Method.CPU_MULTI_THREAD_NUMBA: results["cpu_multi_thread_numba"],
        Method.GPU_CUPY: results["gpu_cupy"],
        Method.GPU_CUSTOM_KERNEL: results["gpu_custom_kernel"],
        Method.GPU_NUMBA: results["gpu_numba"]
    }

    plt.figure(figsize=(12, 8))

    for method in methods_to_plot:
        if method in method_data:
            plt.plot(dimensions, method_data[method], label=method.value, marker='o')
            for i, txt in enumerate(method_data[method]):
                plt.annotate(f'{txt:.6f}', (dimensions[i], method_data[method][i]), textcoords="offset points",
                             xytext=(0, 10), ha='center')

    plt.xscale('log')
    plt.xticks(dimensions, [str(d) for d in dimensions], rotation=45)
    plt.xlabel('Matrix Size (NxN)')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison: Matrix Addition')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("performance_comparison_matrix.png")
    plt.show()

if __name__ == "__main__":
    methods_to_plot = [
        Method.CPU_SINGLE_THREAD,
        Method.CPU_MULTI_THREAD_NUMBA,
        Method.GPU_CUPY,
        Method.GPU_CUSTOM_KERNEL,
        Method.GPU_NUMBA
    ]
    plot_performance(methods_to_plot)
