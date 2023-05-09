#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> // for gettimeofday function
#include <math.h>

#define M 1024 // Number of rows in matrix A
#define K 1024 // Number of columns in matrix A and rows in matrix B
#define N 1024 // Number of columns in matrix B and resulting matrix C

void matrixMul(float *A, float *B, float *C)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    float *h_A, *h_B, *h_C; // Host matrices

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate memory for host matrices
    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C = (float *)malloc(size_C);

    // Initialize host matrices
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            h_A[i * K + j] = i + j;
        }
    }
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_B[i * N + j] = i - j;
        }
    }

    // Measure start time
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Perform matrix multiplication
    matrixMul(h_A, h_B, h_C);

    // Measure end time
    gettimeofday(&end, NULL);

    // Calculate elapsed time in milliseconds
    double elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;

    printf("Matrix multiplication successful! Elapsed time: %.3f ms\n", elapsedTime);

    // Verify result
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            if (fabs(h_C[i * N + j] - sum) > 1e-5)
            {
                printf("Error: mismatch at (%d, %d): expected %f, actual %f\n",
                       i, j, sum, h_C[i * N + j]);
                return -1;
            }
        }
    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
