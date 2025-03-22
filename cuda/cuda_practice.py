from numba import cuda, njit, prange
import numpy as np
import cv2
import time


@cuda.jit
def print_hello_gpu():
    print("Hello World from the GPU")

def hello_gpu():
    # Launch the kernel
    print_hello_gpu[1, 1]()

    # Synchronize to ensure the GPU prints before the script exits
    cuda.synchronize()

@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1

def increment_by_one_gpu():
    #Create input arrays
    N = 1024
    a = np.ones(N, dtype=np.float32)
    multiplier = 1
    

    threadsperblock = cuda.get_current_device().WARP_SIZE * multiplier # 32 in most cases
    max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK # 1024 in most cases
    threadsperblock = max_threads_per_block if threadsperblock > max_threads_per_block else threadsperblock # make sure multiplier is not too big
    blockspergrid = (a.size + (threadsperblock - 1)) // threadsperblock
    increment_by_one[blockspergrid, threadsperblock](a)
    print(a)




@cuda.jit
def grayscale_gpu(img, out):
    x, y = cuda.grid(2)  # Get global thread indices
    
    if x < img.shape[1] and y < img.shape[0]:  # Bounds check
        r, g, b = img[y, x]  # Read RGB
        out[y, x] = 0.299 * r + 0.587 * g + 0.114 * b  # Grayscale formula

def grayscale():   
    # Load image (as RGB)
    img = cv2.imread("image.jpg")
    img = cv2.resize(img, (8000, 8000))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    # **1. Run on CPU**
    start_cpu = time.time()
    gray_cpu = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    end_cpu = time.time()

    # **2. Run on GPU**
    d_img = cuda.to_device(np.ascontiguousarray(img))  # Ensures faster transfer
    d_out = cuda.device_array((h, w), dtype=np.uint8)

    threads_per_block = (32, 32)
    blocks_per_grid = ((w + 15) // 16, (h + 15) // 16)

    start_gpu = time.time()
    grayscale_gpu[blocks_per_grid, threads_per_block](d_img, d_out)
    cuda.synchronize()  # Ensure GPU is done
    end_gpu = time.time()

    gray_gpu = d_out.copy_to_host()  # Copy result back

    # **3. Print Execution Times**
    print(f"CPU Time: {end_cpu - start_cpu:.5f} sec")
    print(f"GPU Time: {end_gpu - start_gpu:.5f} sec")



import cv2
import numpy as np
import time
from numba import cuda

# Function to generate Gaussian kernel
def gaussian_kernel(kernel_size=5, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
                    np.exp(-((x - (kernel_size // 2)) ** 2 + (y - (kernel_size // 2)) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    return kernel / kernel.sum()

# Function to apply Gaussian blur on CPU
def gaussian_blur_cpu(img, kernel, kernel_size):
    img_height, img_width = img.shape
    half_kernel = kernel_size // 2
    out_img = np.zeros_like(img, dtype=np.float32)
    
    for y in range(img_height):
        for x in range(img_width):
            blur_value = 0
            for ky in range(-half_kernel, half_kernel + 1):
                for kx in range(-half_kernel, half_kernel + 1):
                    pixel_x = min(max(x + kx, 0), img_width - 1)  # Handle boundary
                    pixel_y = min(max(y + ky, 0), img_height - 1)
                    blur_value += img[pixel_y, pixel_x] * kernel[ky + half_kernel, kx + half_kernel]
            out_img[y, x] = blur_value
    return out_img

# GPU kernel for Gaussian blur
@cuda.jit
def gaussian_blur_kernel(d_img, d_out, kernel, img_width, img_height, kernel_size):
    x, y = cuda.grid(2)
    
    if x < img_width and y < img_height:
        blur_value = 0
        half_kernel = kernel_size // 2
        
        for ky in range(-half_kernel, half_kernel + 1):
            for kx in range(-half_kernel, half_kernel + 1):
                pixel_x = min(max(x + kx, 0), img_width - 1)
                pixel_y = min(max(y + ky, 0), img_height - 1)
                blur_value += d_img[pixel_y, pixel_x] * kernel[ky + half_kernel, kx + half_kernel]
        
        d_out[y, x] = blur_value

def gaussian_blur():
    # Load image and convert to grayscale
    img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
    img_height, img_width = img.shape

    # Generate Gaussian kernel
    kernel_size = 5
    sigma = 1.0
    kernel = gaussian_kernel(kernel_size, sigma)

    # --- CPU Processing ---
    start_cpu = time.time()
    out_img_cpu = gaussian_blur_cpu(img, kernel, kernel_size)
    cpu_time = time.time() - start_cpu

    # --- GPU Processing ---
    d_img = cuda.to_device(img)
    d_out = cuda.device_array_like(img)

    threads_per_block = (16, 16)
    blocks_per_grid = (img_width + 15) // 16, (img_height + 15) // 16

    start_gpu = time.time()
    gaussian_blur_kernel[blocks_per_grid, threads_per_block](d_img, d_out, kernel, img_width, img_height, kernel_size)
    cuda.synchronize()  # Wait for the GPU to finish
    gpu_time = time.time() - start_gpu

    # Copy result back to CPU
    out_img_gpu = d_out.copy_to_host()

    # Save results
    cv2.imwrite("output_cpu_blurred.jpg", out_img_cpu)
    cv2.imwrite("output_gpu_blurred.jpg", out_img_gpu)

    # --- Output the timings ---
    print(f"CPU Time: {cpu_time:.5f} sec")
    print(f"GPU Time: {gpu_time:.5f} sec")


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

def matrix_multiplication_v2():
    # Define the GPU kernel for matrix multiplication
    @cuda.jit
    def matmul_kernel(A, B, C, N):
        # Thread's absolute position in the grid
        row, col = cuda.grid(2)
        
        if row < N and col < N:
            # Compute a single element in the result matrix C
            temp = 0.0
            for i in range(N):
                temp += A[row, i] * B[i, col]
            C[row, col] = temp

    # CPU matrix multiplication using NumPy
    dim = 15000
    A = np.random.randn(dim, dim).astype(np.float32)
    B = np.random.randn(dim, dim).astype(np.float32)

    # CPU execution
    start_time = time.time()
    C_cpu = np.matmul(A, B)  # Matrix multiplication using NumPy
    elapsed_time = time.time() - start_time
    print('CPU_time = ', elapsed_time)

    # GPU matrix multiplication
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.device_array((dim, dim), dtype=np.float32)

    # GPU execution
    threads_per_block = (32, 32)
    blocks_per_grid = (dim + 15) // 16, (dim + 15) // 16

    start_time = time.time()
    matmul_kernel[blocks_per_grid, threads_per_block](A_device, B_device, C_device, dim)
    cuda.synchronize()  # Wait for the GPU to finish
    elapsed_time = time.time() - start_time
    print('GPU_time = ', elapsed_time)

def matrix_multiplication_v3():
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    from numba import jit, cuda
    import matplotlib.pyplot as plt

    # Matrix multiplication for CPU using Numba JIT
    @jit(nopython=True)
    def matriz_mult_cpu(A, B):
        result = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(B.shape[1]):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    # Matrix multiplication for GPU using Numba CUDA JIT
    @cuda.jit
    def matriz_mult_gpu(A, B, C):
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

    # Matrix sizes to test
    sizes = [100, 200, 500, 1000, 1500, 2000, 2500]

    # CPU and GPU times
    times_cpu = []
    times_gpu = []

    for N in sizes:
        # Create arrays of size NxN
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        C_cpu = np.zeros_like(A)

        # CPU Execution
        start_time_cpu = time.time()
        C_cpu = matriz_mult_cpu(A, B)
        end_time_cpu = time.time()

        # GPU Execution
        A_gpu = cuda.to_device(A)  # Transfer to GPU
        B_gpu = cuda.to_device(B)  # Transfer to GPU
        C_gpu = cuda.device_array((N, N), dtype=np.float32)  # Create result on device

        # Configure the grid and block
        threadsperblock = (16, 16)
        blockspergrid_x = (A.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (B.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # GPU Execution
        start_time_gpu = time.time()
        matriz_mult_gpu[blockspergrid, threadsperblock](A_gpu, B_gpu, C_gpu)
        cuda.synchronize()  # Ensure GPU computation is complete
        end_time_gpu = time.time()

        times_cpu.append(end_time_cpu - start_time_cpu)
        times_gpu.append(end_time_gpu - start_time_gpu)

    # Plot the times in a chart
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_cpu, marker='o', label='CPU')
    plt.plot(sizes, times_gpu, marker='o', label='GPU')
    plt.title('Execution Time for Matrix Multiplication')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()
