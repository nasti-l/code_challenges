from numba import cuda, njit, prange, vectorize
import numpy as np
import math
import time

def matrix_multiplication():
    @njit(parallel=True)  # Runs on CPU, uses multiple threads
    def matrix_multiply_cpu(A, B, C, N):
        for row in prange(N):  # Parallel over rows
            for col in range(N):
                sum_value = 0
                for k in range(N):
                    sum_value += A[row, k] * B[k, col]
                C[row, col] = sum_value
        return C
        
    @cuda.jit
    def matrix_multiply_cuda(A, B, C, N):
        # Get thread indices
        row, col = cuda.grid(2)
        # Compute only if within matrix bounds
        if row < N and col < N:
                sum_value = 0
                for k in range(N):
                    sum_value += A[row, k] * B[k, col]
                C[row, col] = sum_value

    # Example matrix size (you can scale this up for better comparison)
    N = 2  # 1024 x 1024 matrices
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros_like(A) # Allocate memory on the device for result matrix

    print("A shape:", A.shape)
    print("B shape:", B.shape)
    print("C shape:", C.shape)
    # Copy matrices A and B to the GPU
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)


    # CPU execution
    start_cpu = time.time()
    C_cpu = matrix_multiply_cpu(A, B, C, N)
    cpu_time = time.time() - start_cpu
    print(f"CPU Time: {cpu_time:.5f} sec")

    # Prepare grid/block sizes
    threads_per_block = (32, 32)
    blocks_per_grid = (N + threads_per_block[0]-1) // 32, (N + threads_per_block[1]-1) // 32

    # GPU execution
    start_gpu = time.time()
    matrix_multiply_cuda[blocks_per_grid, threads_per_block](d_A, d_B, d_C, N)
    cuda.synchronize()  # Wait for the kernel to finish
    gpu_time = time.time() - start_gpu

    # Copy result back to host
    C_gpu = d_C.copy_to_host()
    print(C_gpu)
    print(C_cpu)

    print(f"GPU Time: {gpu_time:.5f} sec")

def vectorize_guassian_pdf():
    # Precompute the constant
    SQRT_2PI = np.float32(math.sqrt(2 * math.pi))

    # CPU Version
    @vectorize(['float32(float32, float32, float32)'], target='cpu')
    def gaussian_pdf_cpu(x, mean, sigma):
        return math.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * SQRT_2PI)

    # GPU Version
    @vectorize(['float32(float32, float32, float32)'], target='cuda')
    def gaussian_pdf_gpu(x, mean, sigma):
        return math.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * SQRT_2PI)

    # Generate a large dataset
    size = 100_000_000  # 10 million values
    x_values = np.random.uniform(-5, 5, size).astype(np.float32)
    mean = 0.0
    sigma = 1.0

    # Measure CPU Execution Time
    start_cpu = time.time()
    result_cpu = gaussian_pdf_cpu(x_values, mean, sigma)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # Measure GPU Execution Time
    start_gpu = time.time()
    result_gpu = gaussian_pdf_gpu(x_values, mean, sigma)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    print(f"CPU Time: {cpu_time:.5f} sec")
    print(f"GPU Time: {gpu_time:.5f} sec")

    if gpu_time > 0:  # Ensure we don't divide by zero
        speedup = cpu_time / gpu_time
        if speedup >= 1:
            print(f"GPU is {speedup:.2f}x faster than CPU! 🚀")
        else:
            print(f"CPU is {1/speedup:.2f}x faster than GPU! 🤔")
    else:
        print("Error: GPU execution time is zero or invalid!")

vectorize_guassian_pdf()
# matrix_multiplication()
