import cupy as cp
import numpy as np
from numba import cuda
import time


def gpu_vector_addition_cupy(n):
    # Generate two large vectors
    a = cp.random.rand(n)
    b = cp.random.rand(n)
    c = cp.empty_like(a)

    # Start timing
    start_time = time.perf_counter()

    cp.add(a, b, c)

    # Synchronize the device to ensure all operations are finished
    cp.cuda.Device().synchronize()

    # End timing
    end_time = time.perf_counter()

    # Calculate the time taken
    time_taken = end_time - start_time

    return time_taken


def gpu_vector_addition_cupy_custom_kernel(n):
    # Generate two large vectors
    a = cp.random.rand(n)
    b = cp.random.rand(n)

    # Define the custom CUDA kernel for vector addition
    vector_addition_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void vector_addition(const float* a, const float* b, float* c, const int n) {
        int i = threadIdx.x;
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
    ''', 'vector_addition')

    # Allocate memory for the result vector
    c = cp.empty_like(a)

    # Determine the number of threads and blocks
    # Query device properties
    device = cp.cuda.Device()
    max_threads_per_block = device.attributes['MaxThreadsPerBlock']
    max_blocks_per_grid = device.attributes['MaxGridDimX']

    blocks_per_grid = (n + max_threads_per_block - 1) // max_blocks_per_grid

    # Start timing
    start_time = time.perf_counter()

    # Launch the custom kernel
    vector_addition_kernel((blocks_per_grid,), (max_threads_per_block,), (a, b, c, n))

    # Synchronize the device to ensure all operations are finished
    cp.cuda.Device().synchronize()

    # End timing
    end_time = time.perf_counter()

    # Calculate the time taken
    time_taken = end_time - start_time

    return time_taken


@cuda.jit
def vector_add_gpu(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]


def gpu_vector_addition_numba(n):
    a = np.random.rand(n)
    b = np.random.rand(n)

    # Allocate GPU memory and copy data to GPU
    a_device = cuda.to_device(a)
    b_device = cuda.to_device(b)
    c_device = cuda.device_array_like(a)

    # Configure the blocks and threads per block
    threads_per_block = 256
    blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block

    # Start timing
    start_time = time.perf_counter()

    vector_add_gpu[blocks_per_grid, threads_per_block](a_device, b_device, c_device)
    cuda.synchronize()

    # End timing
    end_time = time.perf_counter()

    # Calculate the time taken
    time_taken = end_time - start_time

    return time_taken
