#include <stdio.h>
#include <cuda_runtime.h>

#define M 1024   // Number of rows in matrix A
#define K 1024   // Number of columns in matrix A and rows in matrix B
#define N 1024   // Number of columns in matrix B and resulting matrix C

// CUDA kernel function to perform matrix multiplication
__global__ void matrixMul(float* A, float* B, float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

int main()
{
    float *h_A, *h_B, *h_C;  // Host matrices
    float *d_A, *d_B, *d_C;  // Device matrices

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate memory for host matrices
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    // Initialize host matrices
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = i + j;
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = i - j;
        }
    }

    // Allocate memory for device matrices
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy host matrices to device matrices
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA event handles for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Launch kernel to perform matrix multiplication
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // Record stop time
    cudaEventRecord(stop);

    // Wait for completion of kernel execution
    cudaEventSynchronize(stop);

    // Calculate elapsed time in seconds
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            if (fabs(h_C[i * N + j] - sum) > 1e-5) {
                printf("Error: mismatch at (%d, %d): expected %f, actual %f\n",
                       i, j, sum, h_C[i * N + j]);
                return -1;
            }
        }
    }

    printf("Matrix multiplication successful! Elapsed time: %.3f ms\n", elapsedTime);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
