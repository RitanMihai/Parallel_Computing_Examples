import numpy as np
import time
from numba import njit, prange

def matrix_add_single_thread(A, B, C):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = A[i, j] + B[i, j]

@njit(parallel=True)
def matrix_add_multi_thread(A, B, C):
    for i in prange(A.shape[0]):
        for j in prange(A.shape[1]):
            C[i, j] = A[i, j] + B[i, j]

def cpu_matrix_addition_single_thread(n):
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    C = np.empty_like(A)

    # Start timing
    start_time = time.perf_counter()

    matrix_add_single_thread(A, B, C)

    # End timing
    end_time = time.perf_counter()

    return C, end_time - start_time

def cpu_matrix_addition_multithread_numba(n):
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    C = np.empty_like(A)

    # Start timing
    start_time = time.perf_counter()

    matrix_add_multi_thread(A, B, C)

    # End timing
    end_time = time.perf_counter()

    return C, end_time - start_time
