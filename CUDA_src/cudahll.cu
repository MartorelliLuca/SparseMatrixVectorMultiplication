#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>

#include "../CUDA_include/cudahll.h"
#include "../src/data_structures/hll_matrix.h"
#include "../src/data_structures/performance.h"
#include "../CUDA_include/cuda_hll_kernel_v1.cuh"

float invoke_kernel_1(HLL_matrix *hll_matrix, double *x, double *z)
{
    HLL_matrix d_A;
    double *d_x, *d_y;
    float time;

    // Allocazione memoria su device
    cudaMalloc(&d_A.offsets, hll_matrix->offsets_num * sizeof(int));
    cudaMalloc(&d_A.col_index, hll_matrix->data_num * sizeof(int));
    cudaMalloc(&d_A.data, hll_matrix->data_num * sizeof(double));
    cudaMalloc(&d_A.max_nzr, hll_matrix->hacks_num * sizeof(int));

    cudaMalloc(&d_x, hll_matrix->N * sizeof(double));
    cudaMalloc(&d_y, hll_matrix->M * sizeof(double));

    // Copia dei dati da host a device
    cudaMemcpy(d_A.offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A.col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A.data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A.max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice);

    // Configurazione dei thread e lancio del kernel
    int blockSize = 256;
    int gridSize = (hll_matrix->M + blockSize - 1) / blockSize;

    // Eventi CUDA per la misurazione del tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Inizio misurazione
    hll_matvec_kernel<<<gridSize, blockSize>>>(d_A, d_x, d_y);
    cudaEventRecord(stop); // Fine misurazione

    cudaEventSynchronize(stop);

    // Calcolo del tempo di esecuzione in millisecondi
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copia del risultato dal device all'host
    cudaMemcpy(z, d_y, hll_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Milliseconds = %.16lf\n", milliseconds);

    // Calcolo delle prestazioni in MFLOPS
    time = milliseconds / 1000.0;

    // Deallocazione memoria device e eventi CUDA
    cudaFree(d_A.offsets);
    cudaFree(d_A.col_index);
    cudaFree(d_A.data);
    cudaFree(d_A.max_nzr);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}
