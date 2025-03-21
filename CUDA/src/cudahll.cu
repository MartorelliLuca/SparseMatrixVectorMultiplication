#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>

#include "../../src/data_structures/hll_matrix.h"
#include "../../src/data_structures/performance.h"
#include "../kernel/hll/cuda_hll_kernel_v1.cuh"

double invoke_kernel_1(HLL_matrix *hll_matrix, double *x, double *z)
{
    cudaError_t error;
    int *d_offsets, *d_col_index;
    double *d_data, *d_x, *d_y;
    float time = 0.0;

    // Eventi CUDA per la misurazione del tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocazione memoria GPU con controlli errori migliorati
    if (cudaMalloc((void **)&d_offsets, hll_matrix->offsets_num * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void **)&d_col_index, hll_matrix->data_num * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void **)&d_data, hll_matrix->data_num * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&d_x, hll_matrix->N * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&d_y, hll_matrix->M * sizeof(double)) != cudaSuccess)
    {
        error = cudaGetLastError();
        printf("Error during cudaMalloc!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    // Copia dati da host a device
    cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice);

    // Configurazione kernel
    int blockSize = 32;
    int numBlocks = (hll_matrix->M + blockSize - 1) / blockSize;

    // Avvio misurazione tempo
    cudaEventRecord(start);

    // Esecuzione del kernel CUDA
    hll_kernel_v1<<<numBlocks, blockSize>>>(d_offsets, d_col_index, d_data, d_x, d_y, hll_matrix->M);

    // Sincronizzazione e controllo errori
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Error during kernel execution!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    // Stop misurazione tempo
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // Copia risultati da device a host
    cudaMemcpy(z, d_y, hll_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);

    // Libera memoria GPU
    cudaFree(d_offsets);
    cudaFree(d_col_index);
    cudaFree(d_data);
    cudaFree(d_x);
    cudaFree(d_y);

    // Distruggi eventi CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time / 1000.0; // Convertiamo da millisecondi a secondi
}
