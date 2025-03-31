#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../CUDA_include/cudacsr.h"
#include "../CUDA_include/csr/cuda_csr_kernel_v1.cuh"
#include "../CUDA_include/csr/cuda_csr_kernel_v2.cuh"
#include "../CUDA_include/csr/cuda_csr_kernel_v3.cuh"
#include "../CUDA_include/csr/cuda_csr_kernel_v4.cuh"
// #include "../CUDA_include/csr/cuda_csr_kernel_v5.cuh"

#define CHECK_CUDA_ERROR(call)                                    \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/*Kernel 1*/
float invoke_kernel_csr_1(CSR_matrix *csr_matrix, double *x, double *z, int num_threads_per_block)
{
    int *d_IRP, *d_JA;
    double *d_x, *d_y, *d_AS;
    float time;

    CHECK_CUDA_ERROR(cudaMalloc(&d_IRP, (csr_matrix->M + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_JA, csr_matrix->non_zero_values * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_AS, csr_matrix->non_zero_values * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, csr_matrix->N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, csr_matrix->M * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_IRP, csr_matrix->IRP, (csr_matrix->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_JA, csr_matrix->JA, csr_matrix->non_zero_values * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_AS, csr_matrix->AS, csr_matrix->non_zero_values * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, csr_matrix->N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_y, 0, csr_matrix->M * sizeof(double)));

    printf("Configurazione kernel 1:\n");
    printf("Matrice CSR: %s\n", csr_matrix->name);
    printf("Dimensioni: M = %d, N = %d\n", csr_matrix->M, csr_matrix->N);
    printf("Valori non zero: %d\n", csr_matrix->non_zero_values);

    dim3 blockDim1(WARP_SIZE, num_threads_per_block);
    dim3 gridDim1((csr_matrix->M + blockDim1.y - 1) / blockDim1.y);

    printf("\ncsr_matrix->M = %d\n", csr_matrix->M);
    printf("gridSize = %d\n", gridDim1);
    printf("blockSize = %d\n", blockDim1.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // primo parametro numero warp da usare = grandezza del blocco
    // secondo parametro numero di thread per warp
    // terzo parametro shared memory

    csr_matvec_kernel<<<gridDim1, num_threads_per_block>>>(d_IRP, d_JA, d_AS, csr_matrix->M, d_x, d_y);
    // 32   -> 22.34
    // 16   -> 42.55
    // 8    -> 56.01
    // 4    -> 36.28
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(z, d_y, csr_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Milliseconds = %.16lf\n", milliseconds);

    time = milliseconds / 1000.0;

    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

/*Kernel 2*/
float invoke_kernel_csr_2(CSR_matrix *csr_matrix, double *x, double *z, int num_threads_per_block)
{

    cudaEvent_t start, stop;
    float elapsedTime;
    double *d_x, *d_y;

    CSR_matrix d_csr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.IRP, (csr_matrix->M + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.JA, csr_matrix->non_zero_values * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.AS, csr_matrix->non_zero_values * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, csr_matrix->N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, csr_matrix->M * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.IRP, csr_matrix->IRP, (csr_matrix->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.JA, csr_matrix->JA, csr_matrix->non_zero_values * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.AS, csr_matrix->AS, csr_matrix->non_zero_values * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, csr_matrix->N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_y, 0, csr_matrix->M * sizeof(double)));

    // Configurazione blocchi e griglia kernel
    dim3 blockDim2(WARP_SIZE, num_threads_per_block);
    dim3 gridDim2((csr_matrix->M + blockDim2.y - 1) / blockDim2.y);
    printf("Configurazione kernel 2:\n");
    printf("\ncsr_matrix->M = %d\n", csr_matrix->M);
    printf("blockDim2.y = %d\n", blockDim2.y);
    printf("gridDim2 = %d\n", gridDim2);
    size_t sharedMemSize = blockDim2.y * WARP_SIZE * sizeof(double);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    csr_matvec_shfl_reduction<<<gridDim2, blockDim2, sharedMemSize>>>(d_csr, csr_matrix->M, d_x, d_y);
    // 32   -> 42.73
    // 16   -> 50.76
    // 8    -> 53.87
    // 4    -> 58.54
    cudaDeviceSynchronize(); // Assicura il completamento dell'esecuzione del kernel
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    printf("\nTempo di esecuzione: %.10f ms\n", elapsedTime);

    // Risultato da GPU a CPU
    CHECK_CUDA_ERROR(cudaMemcpy(z, d_y, csr_matrix->M * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_csr.IRP);
    cudaFree(d_csr.JA);
    cudaFree(d_csr.AS);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime / 1000; // Converte ms in secondi
}

/*Kernel 3*/
float invoke_kernel_csr_3(CSR_matrix *csr_matrix, double *x, double *z, int num_threads_per_block)
{
    // 1. Allocazione memoria sulla GPU per i dati
    double *d_x, *d_y;
    CSR_matrix d_csr;
    cudaEvent_t start, stop;
    float elapsedTime;

    // Allocazione della memoria per il vettore x e y
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, csr_matrix->M * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, csr_matrix->M * sizeof(double)));

    // Allocazione memoria per la matrice CSR (dati necessari)
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.IRP, (csr_matrix->M + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.JA, csr_matrix->non_zero_values * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.AS, csr_matrix->non_zero_values * sizeof(double)));

    // Copia dei dati dalla memoria host a quella device
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, csr_matrix->M * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.IRP, csr_matrix->IRP, (csr_matrix->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.JA, csr_matrix->JA, csr_matrix->non_zero_values * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.AS, csr_matrix->AS, csr_matrix->non_zero_values * sizeof(double), cudaMemcpyHostToDevice));

    // 2. Definire il numero di blocchi e thread per blocco
    dim3 blockDim3(WARP_SIZE, num_threads_per_block);
    dim3 gridDim3((csr_matrix->M + blockDim3.y - 1) / blockDim3.y);
    printf("Configurazione kernel 3:\n");
    printf("\ncsr_matrix->M = %d\n", csr_matrix->M);
    printf("blockDim3.y = %d\n", blockDim3.y);
    printf("gridDim3 = %d\n", gridDim3);

    // Allocazione memoria condivisa
    size_t shared_mem_size = num_threads_per_block * WARP_SIZE * sizeof(double);

    // 3. Avvia il kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    csr_matvec_shared_memory<<<gridDim3, blockDim3, shared_mem_size>>>(d_csr, csr_matrix->M, d_x, d_y);

    // 32   -> 36.43
    // 16   -> 40.06
    // 8    -> 47.91
    // 4    -> 50.82

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(z, d_y, csr_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);
    // printf("\n\n");
    // print_vector4(z, 4, "z");
    // printf("\n\n");
    printf("Milliseconds = %.16lf\n", milliseconds);

    elapsedTime = milliseconds / 1000.0;
    // 7. Deallocazione memoria GPU
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_csr.IRP);
    cudaFree(d_csr.JA);
    cudaFree(d_csr.AS);

    return elapsedTime;
}

/*Kernel 4*/
float invoke_kernel_csr_4(CSR_matrix *csr_matrix, double *x, double *z, int num_threads_per_block)
{
    cudaEvent_t start, stop;
    float elapsedTime;
    double *d_x, *d_y;
    CSR_matrix d_csr;

    // Allocazione memoria sulla GPU con gestione errori
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.IRP, (csr_matrix->M + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.JA, csr_matrix->non_zero_values * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.AS, csr_matrix->non_zero_values * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, csr_matrix->N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, csr_matrix->M * sizeof(double)));

    // Copia dati dalla CPU alla GPU con gestione errori
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.IRP, csr_matrix->IRP, (csr_matrix->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.JA, csr_matrix->JA, csr_matrix->non_zero_values * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.AS, csr_matrix->AS, csr_matrix->non_zero_values * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, csr_matrix->N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_y, 0, csr_matrix->M * sizeof(double)));

    // Configurazione blocchi e griglia kernel
    dim3 blockDim4(WARP_SIZE, num_threads_per_block);
    dim3 gridDim4((csr_matrix->M + blockDim4.y - 1) / blockDim4.y);
    printf("Configurazione kernel 4:\n");
    printf("\ncsr_matrix->M = %d\n", csr_matrix->M);
    printf("blockDim4.y = %d\n", blockDim4.y);
    printf("gridDim4 = %d\n", gridDim4);

    // Creazione e avvio degli eventi CUDA con gestione errori
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    size_t sharedMemSize = blockDim4.y * WARP_SIZE * sizeof(double);
    csr_matvec_warp_cacheL2<<<gridDim4, blockDim4, sharedMemSize>>>(d_csr, csr_matrix->M, d_x, d_y);
    // 32   -> 41.90
    // 16   -> 50.68
    // 8    -> 53.63
    // 4    -> 56.06

    cudaDeviceSynchronize(); // Sincronizza GPU con CPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Tempo di esecuzione: %.10f ms\n", elapsedTime);

    // Copia risultato da GPU a CPU con gestione errori
    cudaMemcpy(z, d_y, csr_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_csr.IRP);
    cudaFree(d_csr.JA);
    cudaFree(d_csr.AS);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime / 1000; // Converte ms in secondi
}
