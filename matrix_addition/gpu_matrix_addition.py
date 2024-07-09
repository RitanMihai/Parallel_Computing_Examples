import math
import cupy as cp
import numpy as np
from numba import cuda
import time

def gpu_matrix_addition_cupy(n):
    # Generate two large matrices
    A_gpu = cp.random.rand(n, n).astype(cp.float32)
    B_gpu = cp.random.rand(n, n).astype(cp.float32)
    C_gpu = cp.empty_like(A_gpu)

    # Start timing
    start_time = time.perf_counter()
    C_gpu = A_gpu + B_gpu
    cp.cuda.Device().synchronize()
    # End timing
    end_time = time.perf_counter()

    return end_time - start_time

def gpu_matrix_addition_cupy_custom_kernel(n):
    # Generate two large matrices
    A_gpu = cp.random.rand(n, n).astype(cp.float32)
    B_gpu = cp.random.rand(n, n).astype(cp.float32)

    # Define the custom CUDA kernel for matrix addition
    matrix_addition_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void matrix_addition(const float* A, const float* B, float* C, const int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int index = i * N + j;
        if (i < N && j < N) {
            C[index] = A[index] + B[index];
        }
    }
    ''', 'matrix_addition')

    N = A_gpu.shape[0]
    C_gpu = cp.empty_like(A_gpu)
    threads_per_block = (16, 16)
    blocks_per_grid = (math.ceil(N / 16), math.ceil(N / 16))

    # Start timing
    start_time = time.perf_counter()
    matrix_addition_kernel(blocks_per_grid, threads_per_block, (A_gpu, B_gpu, C_gpu, N))
    cp.cuda.Device().synchronize()
    # End timing
    end_time = time.perf_counter()

    return end_time - start_time

@cuda.jit
def matrix_add_gpu(A, B, C):
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        C[i, j] = A[i, j] + B[i, j]

def gpu_matrix_addition_numba(n):
    # Generate two large matrices
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)

    # Allocate GPU memory and copy data to GPU
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.device_array_like(A)

    threads_per_block = (16, 16)
    blocks_per_grid = (math.ceil(A.shape[0] / threads_per_block[0]), math.ceil(A.shape[1] / threads_per_block[1]))

    # Start timing
    start_time = time.perf_counter()
    matrix_add_gpu[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
    cuda.synchronize()
    # End timing
    end_time = time.perf_counter()

    return end_time - start_time
