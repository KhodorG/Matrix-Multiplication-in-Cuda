#include <stdio.h>
#include <openacc.h>
#include <math.h>

#define M 1024   // Number of rows in matrix A
#define K 1024   // Number of columns in matrix A and rows in matrix B
#define N 1024   // Number of columns in matrix B and resulting matrix C

// CUDA kernel function to perform matrix multiplication
void matrixMul(float* A, float* B, float* C)
{
#pragma acc parallel loop collapse(2) present(A[0:M*K], B[0:K*N], C[0:M*N])
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
#pragma acc loop seq
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    float *h_A, *h_B, *h_C;  // Host matrices
    float *d_A, *d_B, *d_C;  // Device matrices

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C =M * N * sizeof(float);

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
    d_A = (float*)acc_malloc(size_A);
    d_B = (float*)acc_malloc(size_B);
    d_C = (float*)acc_malloc(size_C);

    // Copy host matrices to device matrices
    acc_memcpy_to_device(d_A, h_A, size_A);
    acc_memcpy_to_device(d_B, h_B, size_B);

    // Create OpenACC event handles for timing
    acc_event_t start, stop;
    acc_event_create(&start);
    acc_event_create(&stop);

    // Record start time
    acc_event_record(start);

    // Launch kernel to perform matrix multiplication
    matrixMul(d_A, d_B, d_C);

// Record stop time
    acc_event_record(stop);

    // Wait for completion of kernel execution
    acc_event_wait(stop);

    // Calculate elapsed time in seconds
    float elapsedTime;
    acc_event_elapsed_time(start, stop, &elapsedTime);

    // Copy result from device to host
    acc_memcpy_from_device(h_C, d_C, size_C);

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
    acc_free(d_A);
    acc_free(d_B);
    acc_free(d_C);

    return 0;
}
