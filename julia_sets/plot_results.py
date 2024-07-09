import json
import matplotlib.pyplot as plt
from enum import Enum


class Method(Enum):
    CPU_SINGLE_THREAD = "CPU Single Thread"
    CPU_MULTI_THREAD_NUMBA = "CPU Multi Thread Numba"
    GPU_CUPY = "GPU CuPy"
    GPU_NUMBA = "GPU Numba"
    GPU_CUSTOM_KERNEL = "GPU Custom Kernel"


def plot_performance(results):
    resolutions = results["resolutions"]
    num_resolutions = len(resolutions)

    colors = {
        Method.CPU_SINGLE_THREAD: 'r',
        Method.CPU_MULTI_THREAD_NUMBA: 'g',
        Method.GPU_CUPY: 'b',
        Method.GPU_NUMBA: 'm',
        Method.GPU_CUSTOM_KERNEL: 'c'
    }

    markers = {
        Method.CPU_SINGLE_THREAD: 'o',
        Method.CPU_MULTI_THREAD_NUMBA: '^',
        Method.GPU_CUPY: 's',
        Method.GPU_NUMBA: 'd',
        Method.GPU_CUSTOM_KERNEL: 'x'
    }

    fig, axes = plt.subplots(nrows=1, ncols=num_resolutions, figsize=(20, 6), sharey=True)

    for ax, resolution in zip(axes, resolutions):
        width, height = resolution
        resolution_key = f'{width}x{height}'

        for method in Method:
            xs = []
            ys = []

            for entry in results[method.value.lower().replace(" ", "_")]:
                if entry["resolution"] == resolution:
                    iterations = entry["iterations"]
                    time = entry["time"]
                    xs.append(iterations)
                    ys.append(time)

            # Scatter plot
            ax.scatter(xs, ys, c=colors[method], marker=markers[method], label=method.value)

            # Line plot
            ax.plot(xs, ys, c=colors[method])

        ax.set_xlabel('Number of Iterations')
        ax.set_title(f'{resolution_key}')
        ax.grid(True)

    axes[0].set_ylabel('Time (seconds)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(Method))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("performance_comparison_julia_subplots.png")
    plt.show()


if __name__ == "__main__":
    with open("performance_results.json", "r") as f:
        results = json.load(f)
    plot_performance(results)
