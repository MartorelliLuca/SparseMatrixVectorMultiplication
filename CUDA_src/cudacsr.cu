#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../CUDA_include/cudacsr.h"
#include "../CUDA_include/cuda_csr_kernel_v1.cuh"

// qua alloca strutture dati per le chiamate a kernel (partizione del carico di ogni warp)

float invoke_kernel_csr_1(CSR_matrix *csr_matrix, double *x, double *z)
{
    CSR_matrix d_A;
    double *d_x, *d_y;
    float time;

    cudaMalloc(&d_A.IRP, (csr_matrix->M + 1) * sizeof(int));
    cudaMalloc(&d_A.JA, csr_matrix->non_zero_values * sizeof(int));
    cudaMalloc(&d_A.AS, csr_matrix->non_zero_values * sizeof(double));

    cudaMalloc(&d_x, csr_matrix->N * sizeof(double));
    cudaMalloc(&d_y, csr_matrix->M * sizeof(double));

    cudaMemcpy(d_A.IRP, csr_matrix->IRP, (csr_matrix->M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A.JA, csr_matrix->JA, csr_matrix->non_zero_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A.AS, csr_matrix->AS, csr_matrix->non_zero_values * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, x, csr_matrix->N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, csr_matrix->M * sizeof(double));

    int blockSize = 256;
    int gridSize = (csr_matrix->M + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    csr_matvec_kernel<<<gridSize, blockSize>>>(d_A, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(z, d_y, csr_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Milliseconds = %.16lf\n", milliseconds);

    time = milliseconds / 1000.0;

    cudaFree(d_A.IRP);
    cudaFree(d_A.JA);
    cudaFree(d_A.AS);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}
