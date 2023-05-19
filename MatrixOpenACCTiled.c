#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 1024   // Number of rows in matrix A and resulting matrix C
#define K 1024   // Number of columns in matrix A and rows in matrix B
#define N 1024   // Number of columns in matrix B

#define TILE_SIZE 32   // Tile size for tiling

void matrixMulTiled(float* A, float* B, float* C)
{
    #pragma acc data copyin(A[0:M*K], B[0:K*N]) copyout(C[0:M*N])
    {
        #pragma acc kernels loop gang((M - 1) / TILE_SIZE + 1) vector((N - 1) / TILE_SIZE + 1) present(A[0:M*K], B[0:K*N], C[0:M*N])
        for (int by = 0; by < (M - 1) / TILE_SIZE + 1; by++) {
            for (int bx = 0; bx < (N - 1) / TILE_SIZE + 1; bx++) {
                #pragma acc cache(As[0:TILE_SIZE][0:TILE_SIZE], Bs[0:TILE_SIZE][0:TILE_SIZE])
                {
                    float As[TILE_SIZE][TILE_SIZE];
                    float Bs[TILE_SIZE][TILE_SIZE];

                    for (intty = 0; ty < TILE_SIZE; ty++) {
                        for (int tx = 0; tx < TILE_SIZE; tx++) {
                            As[ty][tx] = 0.0f;
                            Bs[ty][tx] = 0.0f;
                        }
                    }

                    int row_start = by * TILE_SIZE;
                    int row_end = (by + 1) * TILE_SIZE;
                    int col_start = bx * TILE_SIZE;
                    int col_end = (bx + 1) * TILE_SIZE;

                    for (int t = 0; t < K; t += TILE_SIZE) {
                        for (int ty = 0; ty < TILE_SIZE; ty++) {
                            int row = row_start + ty;
                            if (row < M) {
                                for (int tx = 0; tx < TILE_SIZE; tx++) {
                                    int col = t + tx;
                                    if (col < K) {
                                        As[ty][tx] = A[row * K + col];
                                    }
                                }
                            }
                        }

                        for (int tx = 0; tx < TILE_SIZE; tx++) {
                            for (int ty = 0; ty < TILE_SIZE; ty++) {
                                int col = col_start + tx;
                                if (col < N) {
                                    Bs[ty][tx] = B[t * N + ty * N + col];
                                }
                            }
                        }

                        #pragma acc barrier

                        for (int i = 0; i < TILE_SIZE; i++) {
                            for (int j = 0; j < TILE_SIZE; j++) {
                                for (int k = 0; k < TILE_SIZE; k++) {
                                    C[(row_start + i) * N + (col_start + j)] += As[i][k] * Bs[k][j];
                                }
                            }
                        }
                    }
                }
            }
        }
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
        for (int j =0; j < N; j++) {
            h_B[i * N + j] = i - j;
        }
    }

    // Allocate memory for device matrices
    #pragma acc enter data create(d_A[0:M*K], d_B[0:K*N], d_C[0:M*N])

    // Copy host matrices to device
    #pragma acc update device(h_A[0:M*K], h_B[0:K*N])

    // Create OpenACC events for timing
    double start_time = omp_get_wtime();
    #pragma acc host_data use_device(d_A, d_B, d_C)
    {
        // Launch kernel for matrix multiplication using tiling
        matrixMulTiled(d_A, d_B, d_C);
    }
    double end_time = omp_get_wtime();

    printf("Matrix multiplication using tiling successful! Elapsed time: %.3f ms\n", (end_time-start_time)*1000.0);

    // Copy result from device to host
    #pragma acc update self(h_C[0:M*N])

    // Verify result
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k *N + j];
            }
            if (fabs(h_C[i * N + j] - sum) > 1e-5) {
                printf("Error: mismatch at (%d, %d): expected %f, actual %f\n",
                       i, j, sum, h_C[i * N + j]);
                return -1;
            }
        }
    }

    // Free memory
    #pragma acc exit data delete(d_A[0:M*K], d_B[0:K*N], d_C[0:M*N])
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
