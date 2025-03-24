#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../CUDA_include/cudacsr.h"
#include "../CUDA_include/cuda_csr_kernel_v1.cuh"

// qua alloca strutture dati per le chiamate a kernel (partizione del carico di ogni warp)

float invoke_kernel_v1(CSR_matrix *csr_matrix, double *x, double *z)
{
    double *d_x, *d_y;
    int *d_row, *d_col;
    double *d_val;
    cudaMalloc(&d_x, csr_matrix->N * sizeof(double));
    cudaMalloc(&d_y, csr_matrix->N * sizeof(double));
    cudaMalloc(&d_row, (csr_matrix->N + 1) * sizeof(int));
    cudaMalloc(&d_col, csr_matrix->non_zero_values * sizeof(int));
    cudaMalloc(&d_val, csr_matrix->non_zero_values * sizeof(double));
    if (d_x == NULL || d_y == NULL || d_row == NULL || d_col == NULL || d_val == NULL)
    {
        printf("Errore nell'allocazione della memoria per il device\n");
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_x, x, csr_matrix->N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, csr_matrix->N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, csr_matrix->IRP, (csr_matrix->N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csr_matrix->JA, csr_matrix->non_zero_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, csr_matrix->AS, csr_matrix->non_zero_values * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (csr_matrix->N + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    csr_matvec_warps_per_row<<<num_blocks, block_size>>>(csr->N, d_row, d_col, d_val, d_x, d_y);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(z, d_y, csr_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Milliseconds = %.16lf\n", milliseconds);

    time = milliseconds / 1000.0;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);

    return time;
}
