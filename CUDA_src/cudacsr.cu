#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../CUDA_include/cudacsr.h"
#include "../CUDA_include/csr/cuda_csr_kernel_v1.cuh"
#include "../CUDA_include/csr/cuda_csr_kernel_v2.cuh"
#include "../CUDA_include/csr/cuda_csr_kernel_v3.cuh"
#include "../CUDA_include/csr/cuda_csr_kernel_v4.cuh"
#include "../CUDA_include/csr/cuda_csr_kernel_v5.cuh"

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
float invoke_kernel_csr_1(CSR_matrix *csr_matrix, double *x, double *z)
{
    CSR_matrix d_csr;
    double *d_x, *d_y;
    float time;

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

    int blockSize = 8;
    int gridSize = (csr_matrix->M + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    csr_matvec_kernel<<<gridSize, blockSize>>>(d_csr, d_x, d_y);
    // 32   -> 22
    // 16   -> 41.8
    // 12   -> 51.13
    // 11   -> 50.85
    // 10   -> 51.96
    // 8    -> 53.41 -> 55.04 -> 58.55
    // 4    -> 35.82
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CHECK_CUDA_ERROR(cudaMemcpy(z, d_y, csr_matrix->M * sizeof(double), cudaMemcpyDeviceToHost));
    printf("Milliseconds = %.16lf\n", milliseconds);

    time = milliseconds / 1000.0;

    cudaFree(d_csr.IRP);
    cudaFree(d_csr.JA);
    cudaFree(d_csr.AS);
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
    dim3 blockDim2(WARP_SIZE, 4);
    printf("\n\nsono nel secondo\n\n\n");

    dim3 gridDim2((csr_matrix->M + blockDim2.y - 1) / blockDim2.y);
    printf("blockDim2.y = %d\n", blockDim2.y);
    printf("gridDim2 = %d\n", blockDim2.y);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    size_t sharedMemSize = blockDim2.y * WARP_SIZE * sizeof(double);
    csr_matvec_warp_shmem<<<gridDim2, blockDim2, sharedMemSize>>>(d_csr, d_x, d_y);
    // 32   -> mala noticia     | 41.39
    // 16   -> mala noticia     | 48.93
    // 8    -> mala noticia     | 52.08
    // 5    -> mala noticia     | 51.52
    // 4    -> mala noticia     | 54.32
    // 3    -> mala noticia     |
    // 2    -> mala noticia     |
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
float invoke_kernel_csr_3(CSR_matrix *csr_matrix, double *x, double *z)
{
    cudaEvent_t start, stop;
    float elapsedTime;
    double *d_x, *d_y;
    CSR_matrix d_csr;

    // Allocazione memoria sulla GPU
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.IRP, (csr_matrix->M + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.JA, csr_matrix->non_zero_values * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_csr.AS, csr_matrix->non_zero_values * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, csr_matrix->N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, csr_matrix->M * sizeof(double)));

    // Copia dati dalla CPU alla GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.IRP, csr_matrix->IRP, (csr_matrix->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.JA, csr_matrix->JA, csr_matrix->non_zero_values * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr.AS, csr_matrix->AS, csr_matrix->non_zero_values * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, csr_matrix->N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_y, 0, csr_matrix->M * sizeof(double)));

    dim3 blockDim3(WARP_SIZE, 4);
    dim3 gridDim3((csr_matrix->M + blockDim3.y - 1) / blockDim3.y);
    printf("\n\nblockDim3.y = %d\n", blockDim3.y);
    printf("gridDim3 = %d\n", blockDim3.y);

    // Creazione e avvio degli eventi CUDA
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    size_t sharedMemSize = blockDim3.y * WARP_SIZE * sizeof(double);
    // 32   -> 39.20
    // 16   -> 46.88
    // 8    -> 49.58
    // 5    -> 48.80
    // 4    -> 50.82
    // 3    -> 49.63

    csr_matvec_atomic_warp_shmem<<<gridDim3, blockDim3, sharedMemSize>>>(d_csr, d_x, d_y);
    cudaDeviceSynchronize(); // Sincronizza GPU con CPU

    // Fine timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Tempo di esecuzione: %.10f ms\n", elapsedTime);

    // Copia risultato da GPU a CPU
    cudaMemcpy(z, d_y, csr_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);

    // Libera memoria GPU
    cudaFree(d_csr.IRP);
    cudaFree(d_csr.JA);
    cudaFree(d_csr.AS);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime / 1000; // Converte ms in secondi
}

/*Kernel 4*/
float invoke_kernel_csr_4(CSR_matrix *csr_matrix, double *x, double *z)
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
    dim3 blockDim4(WARP_SIZE, 8);
    dim3 gridDim4((csr_matrix->M + blockDim4.y - 1) / blockDim4.y);
    printf("Configurazione kernel 4:\n");
    printf(" - blockDim4.y = %d\n", blockDim4.y);
    printf(" - gridDim4 = %d\n", gridDim4.x);

    // Creazione e avvio degli eventi CUDA con gestione errori
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    size_t sharedMemSize = blockDim4.y * WARP_SIZE * sizeof(double);
    csr_matvec_warp_cacheL2<<<gridDim4, blockDim4, sharedMemSize>>>(d_csr, d_x, d_y);
    // 32   -> 41.67         | 41.98
    // 16   -> 48.79         |
    // 8    -> 52.06         | 58.71
    // 5    -> 51.54         |
    // 4    -> 55.44         | 58.62
    // 3    -> 53.41
    // 2    ->

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

// Kernel 5
// float invoke_kernel_csr_5(CSR_matrix *d_csr, double *d_x, double *d_y)
// {
// int *d_queue, *d_queue_index;
// cudaMalloc(&d_queue, d_csr.M * sizeof(int));
// cudaMalloc(&d_queue_index, sizeof(int));
// cudaMemset(d_queue_index, 0, sizeof(int));

// // Inizializza la coda con gli indici di riga
// init_queue<<<(d_csr.M + 255) / 256, 256>>>(d_queue, d_csr.M);

// // Esegui il kernel
// int num_blocks = (d_csr.M + WARP_SIZE - 1) / WARP_SIZE;
// csr_matvec_dynamic<<<num_blocks, 128>>>(d_csr, d_x, d_y, d_queue, d_queue_index);

// cudaFree(d_queue);
// cudaFree(d_queue_index);
// }
