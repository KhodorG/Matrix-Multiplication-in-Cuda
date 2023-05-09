#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 1000 // Number of rows in matrix A
#define K 1000 // Number of columns in matrix A and rows in matrix B
#define N 1000 // Number of columns in matrix B and resulting matrix C

int main()
{
    int **A, **B, **C;         // Define matrices A, B, and C as pointers to pointers
    struct timeval start, end; // Define time variables

    // Allocate memory for matrices A, B, and C
    A = (int **)malloc(M * sizeof(int *));
    B = (int **)malloc(K * sizeof(int *));
    C = (int **)malloc(M * sizeof(int *));
    for (int i = 0; i < M; i++)
    {
        A[i] = (int *)malloc(K * sizeof(int));
        C[i] = (int *)malloc(N * sizeof(int));
    }
    for (int i = 0; i < K; i++)
    {
        B[i] = (int *)malloc(N * sizeof(int));
    }

    // Initialize matrices A and B
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            A[i][j] = i + j;
        }
    }
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            B[i][j] = i - j;
        }
    }

    // Perform matrix multiplication
    gettimeofday(&start, NULL); // Start timer
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int p = 0; p < K; p++)
            {
                sum += A[i][p] * B[p][j];
            }
            C[i][j] = sum;
        }
    }
    gettimeofday(&end, NULL); // End timer

    double time_spent = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0; // Calculate execution time in milliseconds

    // Print the execution time
    printf("Execution time: %.3f ms\n", time_spent);

    // Free memory for matrices A, B, and C
    for (int i = 0; i < M; i++)
    {
        free(A[i]);
        free(C[i]);
    }
    for (int i = 0; i < K; i++)
    {
        free(B[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}
