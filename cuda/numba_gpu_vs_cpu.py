from numba import cuda, njit, prange
import numpy as np
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
matrix_multiplication()