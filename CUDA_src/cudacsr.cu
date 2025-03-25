#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../CUDA_include/cudacsr.h"
#include "../CUDA_include/csr/cuda_csr_kernel_v1.cuh"
#include "../CUDA_include/csr/cuda_csr_kernel_v2.cuh"

// qua alloca strutture dati per le chiamate a kernel (partizione del carico di ogni warp)

/*Kernel 1*/
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

    int blockSize = 8;
    // 256  -> 16
    // 32   -> 22
    // 16   -> 41.8
    // 8    -> 53.41
    // 10   -> 51.96
    // 11   -> 50.85
    // 12   -> 51.13
    // 64   -> 15.40
    // 128  -> 15.61
    // 512  -> 15.38
    // 4    -> 35.82
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

/*Kernel 2*/
float invoke_kernel_csr_2(CSR_matrix *csr_matrix, double *x, double *z)
{
    cudaEvent_t start, stop;
    float elapsedTime;
    double *d_x, *d_y;

    CSR_matrix d_mat;
    cudaMalloc(&d_mat.IRP, (csr_matrix->M + 1) * sizeof(int));
    cudaMalloc(&d_mat.JA, csr_matrix->non_zero_values * sizeof(int));
    cudaMalloc(&d_mat.AS, csr_matrix->non_zero_values * sizeof(double));
    cudaMalloc(&d_x, csr_matrix->N * sizeof(double));
    cudaMalloc(&d_y, csr_matrix->M * sizeof(double));

    cudaMemcpy(d_mat.IRP, csr_matrix->IRP, (csr_matrix->M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat.JA, csr_matrix->JA, csr_matrix->non_zero_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat.AS, csr_matrix->AS, csr_matrix->non_zero_values * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, csr_matrix->N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, csr_matrix->M * sizeof(double));

    // Configurazione blocchi e griglia kernel
    dim3 blockDim(WARP_SIZE, 512);
    // 256  -> mala noticia
    // 32   -> mala noticia
    // 16   -> mala noticia
    // 8    -> mala noticia
    // 64   -> mala noticia
    // 128  -> mala noticia
    // 512  -> mala noticia
    // 5    -> mala noticia
    // 4    -> 50.82
    // 3    -> 49.62

    dim3 gridDim((csr_matrix->M + blockDim.y - 1) / blockDim.y);
    printf("blockDim.y = %d\n", blockDim.y);
    printf("gridDim = %d\n", blockDim.y);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    size_t sharedMemSize = blockDim.y * WARP_SIZE * sizeof(double);
    csr_matvec_warp_shmem<<<gridDim, blockDim, sharedMemSize>>>(d_mat, d_x, d_y);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Tempo di esecuzione: %.10f ms\n", elapsedTime);

    // Risultato da GPU a CPU
    cudaMemcpy(z, d_y, csr_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);

    // Libera memoria sulla GPU
    cudaFree(d_mat.IRP);
    cudaFree(d_mat.JA);
    cudaFree(d_mat.AS);
    cudaFree(d_x);
    cudaFree(d_y);

    return elapsedTime / 1000;
}

// float invoke_kernel_csr_3(CSR_matrix *csr_matrix, double *x, double *z)
// {}