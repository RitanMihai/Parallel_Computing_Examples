from threading import Thread

import numpy as np
import time
import multiprocessing as mp

from numba import njit, prange

# Util methods
def vector_add_single_thread(a, b, c):
    for i in range(a.size):
        c[i] = a[i] + b[i]

def vector_addition_chunk(start, end, a, b, c):
    # c[start:end] = a[start:end] + b[start:end] # <- Numpy optimization
    for i in range(start, end):
        c[i] = a[i] + b[i]

@njit(parallel=True)
def vector_add_multi_thread(a, b, c):
    for i in prange(a.size):
        c[i] = a[i] + b[i]

def cpu_vector_addition_single_thread(n):
    # Generate two large vectors
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.empty_like(a)

    # Start timing
    start_time = time.perf_counter()

    vector_add_single_thread(a, b, c)
    # End timing
    end_time = time.perf_counter()
    # Calculate the time taken
    time_taken = end_time - start_time

    return time_taken
def cpu_vector_addition_vanilla_python(n):
    # Get the number of available CPU cores - 24 on my machine
    num_cores = mp.cpu_count()

    # Generate two large vectors
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.empty_like(a)

    chunk_size = len(a) // num_cores
    threads = []

    # Start timing
    start_time = time.perf_counter()

    for i in range(num_cores):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_cores - 1 else len(a)
        thread = Thread(target=vector_addition_chunk, args=(start, end, a, b, c))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # End timing
    end_time = time.perf_counter()

    time_taken = end_time - start_time
    return time_taken

def cpu_vector_addition_multithread_numba(n):
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.empty_like(a)

    # Start timing
    start_time = time.perf_counter()
    vector_add_multi_thread(a, b, c)
    # End timing
    end_time = time.perf_counter()

    # Calculate the time taken
    time_taken = end_time - start_time

    return time_taken