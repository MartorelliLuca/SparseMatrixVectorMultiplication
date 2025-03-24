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
#include "../CUDA_include/hll/cuda_hll_kernel_v1.cuh"

float invoke_kernel_1(HLL_matrix *hll_matrix, double *x, double *z)
{
    int thread_num = 256;
    int block_num = hll_matrix->hacks_num / thread_num;
    double *data;
    double *d_x, *d_y;
    int *col_indexes, *maxnrz, *offsets;
    float time;

    cudaError_t error;
    cudaEvent_t start, stop;

    // Memory allocation for Device
    error = cudaMalloc(&data, hll_matrix->data_num * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMalloc in invoke kernel 1 for data\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&col_indexes, hll_matrix->data_num * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMalloc in invoke kernel 1 for col indexes\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&maxnrz, hll_matrix->hacks_num * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMalloc in invoke kernel 1 for maxnrz\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&offsets, hll_matrix->offsets_num * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMalloc in invoke kernel 1 for offsets\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&d_x, hll_matrix->data_num * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMalloc in invoke kernel 1 for d_x\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&d_y, hll_matrix->data_num * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMalloc in invoke kernel 1 for d_y\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to device
    error = cudaMemcpy(data, hll_matrix->data, hll_matrix->data_num, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMemcpy in invoke kernel 1 for data\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(col_indexes, hll_matrix->col_index, hll_matrix->data_num, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMemcpy in invoke kernel 1 for col indexes\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(maxnrz, hll_matrix->max_nzr, hll_matrix->hacks_num, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMemcpy in invoke kernel 1 for col indexes\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(offsets, hll_matrix->offsets, hll_matrix->offsets_num, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMemcpy in invoke kernel 1 for col indexes\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_x, x, hll_matrix->offsets_num, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMemcpy in invoke kernel 1 for col indexes\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Configurazione dei thread e lancio del kernel

    // Eventi CUDA per la misurazione del tempo
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // Invocazione del Kernel CUDA
    cuda_kernel_0<<<block_num, thread_num>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num, data, offsets, col_indexes, maxnrz, d_x, hll_matrix->N);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    // Calculation of execution time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results from device to host
    error = cudaMemcpy(z, d_y, hll_matrix->offsets_num, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaMemcpy in invoke kernel 1 for col d_y\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Calculation of performance
    time = milliseconds / 1000;

    // Device memory deallocation and CUDA events
    cudaFree(data);
    cudaFree(col_indexes);
    cudaFree(maxnrz);
    cudaFree(offsets);
    cudaFree(d_y);
    cudaFree(d_x);

    error = cudaDeviceReset();
    if (error != cudaSuccess)
    {
        printf("Error occour in cudaDeviceReset in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}
