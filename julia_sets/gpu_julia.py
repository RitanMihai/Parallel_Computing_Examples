import time
import numpy as np
from numba import cuda
import cupy as cp


@cuda.jit
def julia_set_numba(x_min, x_max, y_min, y_max, c, max_iter, width, height, output):
    x, y = cuda.grid(2)
    if x < width and y < height:
        zx = x_min + x * (x_max - x_min) / width
        zy = y_min + y * (y_max - y_min) / height
        z = complex(zx, zy)
        iteration = 0

        while abs(z) < 2 and iteration < max_iter:
            z = z * z + c
            iteration += 1

        output[y, x] = iteration


def julia_set_numba_with_timing(width, height, x_min, x_max, y_min, y_max, c, max_iter):
    img = np.zeros((height, width), dtype=np.int32)
    d_output = cuda.to_device(img)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Start timing
    start_time = time.perf_counter()

    # Generate the Julia set
    julia_set_numba[blocks_per_grid, threads_per_block](x_min, x_max, y_min, y_max, c, max_iter, width, height,
                                                        d_output)
    cuda.synchronize()

    # End timing
    end_time = time.perf_counter()

    # Calculate the time taken
    time_taken = end_time - start_time

    # Copy result back to host
    img = d_output.copy_to_host()

    return img, time_taken


def julia_set_cupy_custom_kernel(width, height, x_min, x_max, y_min, y_max, c, max_iter):
    kernel_code = '''
    extern "C" __global__ void julia_set(float x_min, float x_max, float y_min, float y_max, float c_real, float c_imag, int max_iter, int width, int height, int* output) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < width && y < height) {
            float zx = x_min + x * (x_max - x_min) / width;
            float zy = y_min + y * (y_max - y_min) / height;
            float2 z = make_float2(zx, zy);
            float2 c = make_float2(c_real, c_imag);
            int iteration = 0;

            while (z.x * z.x + z.y * z.y < 4.0f && iteration < max_iter) {
                float temp = z.x * z.x - z.y * z.y + c.x;
                z.y = 2.0f * z.x * z.y + c.y;
                z.x = temp;
                iteration += 1;
            }

            output[y * width + x] = iteration;
        }
    }
    '''

    module = cp.RawModule(code=kernel_code)
    julia_set_kernel = module.get_function('julia_set')

    img = cp.zeros((height, width), dtype=cp.int32)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    c_real = cp.float32(c.real)
    c_imag = cp.float32(c.imag)

    # Start timing
    start_time = time.perf_counter()

    julia_set_kernel((blocks_per_grid_x, blocks_per_grid_y), (threads_per_block[0], threads_per_block[1]),
                     (cp.float32(x_min), cp.float32(x_max), cp.float32(y_min), cp.float32(y_max),
                      c_real, c_imag, cp.int32(max_iter), cp.int32(width), cp.int32(height), img))

    # Synchronize the device to ensure all operations are finished
    cp.cuda.Device().synchronize()

    # End timing
    end_time = time.perf_counter()

    # Calculate the time taken
    time_taken = end_time - start_time

    return img.get(), time_taken



def julia_set_cupy(width, height, x_min, x_max, y_min, y_max, c, max_iter):
    # Create a grid of complex numbers
    x = cp.linspace(x_min, x_max, width)
    y = cp.linspace(y_min, y_max, height)
    X, Y = cp.meshgrid(x, y)
    Z = X + 1j * Y

    # Initialize the iterations array
    iterations = cp.zeros(Z.shape, dtype=int)

    # Start timing
    start_time = time.perf_counter()

    for i in range(max_iter):
        mask = cp.abs(Z) < 2
        iterations[mask] = i
        Z[mask] = Z[mask] ** 2 + c

    # Synchronize the device to ensure all operations are finished
    cp.cuda.Device().synchronize()

    # End timing
    end_time = time.perf_counter()

    # Calculate the time taken
    time_taken = end_time - start_time

    return iterations.get(), time_taken
