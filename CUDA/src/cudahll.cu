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

double prepare_kernel_v1(HLL_matrix *hll_matrix, double *x, double *z)
{
    cudaError_t error;
    int *d_offsets, *d_col_index;
    double *d_data, *d_x, *d_y;
    double time = 0.0;

    // Create memory on GPU
    if (cudaMalloc((void **)&d_offsets, hll_matrix->offsets_num * sizeof(int)) != cudaSuccess)
    {
        error = cudaGetLastError();
        printf("Error occour in cudaMalloc for d_offset!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void **)&d_col_index, hll_matrix->data_num * sizeof(int)) != cudaSuccess)
    {
        error = cudaGetLastError();
        printf("Error occour in cudaMalloc for d_col_index!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void **)&d_data, hll_matrix->data_num * sizeof(double)) != cudaSuccess)
    {
        error = cudaGetLastError();
        printf("Error occour in cudaMalloc for d_data!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void **)&d_x, hll_matrix->N * sizeof(double)) != cudaSuccess)
    {
        error = cudaGetLastError();
        printf("Error occour in cudaMalloc for d_x!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void **)&d_y, hll_matrix->M * sizeof(double)) != cudaSuccess)
    {
        error = cudaGetLastError();
        printf("Error occour in cudaMalloc for d_y!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    // Copy data from host to device
    if (cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        error = cudaGetLastError();
        printf("Error occour in cudaMemcpy for d_offset!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        error = cudaGetLastError();
        printf("Error occour in cudaMemcpy for d_col_index!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        error = cudaGetLastError();
        printf("Error occour in cudaMemcpy for d_data!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        error = cudaGetLastError();
        printf("Error occour in cudaMalloc for d_x!\nError code: %d\n", error);
        exit(EXIT_FAILURE);
    }

    // Kernel CUDA Configuration
    int blockSize = 32;
    int numBlocks = (hll_matrix->M + blockSize - 1) / blockSize;

    // Run Kernel CUDA
    hll_kernel_v1<<<numBlocks, blockSize>>>(d_offsets, d_col_index, d_data, d_x, d_y, hll_matrix->M);
    cudaDeviceSynchronize();

    // Copy of result
    cudaMemcpy(z, d_y, hll_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_offsets);
    cudaFree(d_col_index);
    cudaFree(d_data);
    cudaFree(d_x);
    cudaFree(d_y);

    return time;
}