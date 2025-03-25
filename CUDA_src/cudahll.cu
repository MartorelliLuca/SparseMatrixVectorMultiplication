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
#include "../CUDA_include/hll/cuda_hll_kernel_v2.cuh"

float invoke_kernel_1(HLL_matrix *hll_matrix, double *x, double *z, int num_threads)
{
    int block_num = (hll_matrix->hacks_num + num_threads - 1) / num_threads;

    double *d_data, *d_x, *d_y;
    int *d_col_index, *d_max_nzr, *d_offsets;
    cudaEvent_t start, stop;
    cudaError_t error;

    error = cudaMalloc(&d_data, hll_matrix->data_num * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda malloc for d_data in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&d_col_index, hll_matrix->data_num * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda malloc for d_col_index in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&d_max_nzr, hll_matrix->hacks_num * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda malloc for d_max_nzr in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&d_offsets, hll_matrix->offsets_num * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda malloc for d_offsets in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&d_x, hll_matrix->N * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda malloc for d_max_nzr in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&d_y, hll_matrix->M * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda malloc for d_y in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda memcpy for d_data in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda memcpy for d_col_index in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda memcpy for d_max_nzr in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda memcpy for d_offsets in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda memcpy for d_x in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cuda_kernel_1<<<block_num, num_threads>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num, d_data, d_offsets, d_col_index, d_max_nzr, d_x, hll_matrix->M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    error = cudaMemcpy(z, d_y, hll_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        printf("Error occour in cuda memcpy for d_y in invoke kernel 1\nError: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_data);
    cudaFree(d_col_index);
    cudaFree(d_max_nzr);
    cudaFree(d_offsets);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 1000.0;
}

float invoke_kernel_2(HLL_matrix *hll_matrix, double *x, double *z, int num_threads)
{
    int M = hll_matrix->M;
    int block_num = (hll_matrix->hacks_num + num_threads - 1) / num_threads;

    double *d_data, *d_x, *d_y;
    int *d_col_index, *d_max_nzr, *d_offsets;
    cudaMalloc(&d_data, hll_matrix->data_num * sizeof(double));
    cudaMalloc(&d_col_index, hll_matrix->data_num * sizeof(int));
    cudaMalloc(&d_max_nzr, hll_matrix->hacks_num * sizeof(int));
    cudaMalloc(&d_offsets, hll_matrix->offsets_num * sizeof(int));
    cudaMalloc(&d_x, hll_matrix->N * sizeof(double));
    cudaMalloc(&d_y, hll_matrix->M * sizeof(double));

    cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemset(d_y, 0, hll_matrix->M * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int shared_mem_size = num_threads * sizeof(double);
    cuda_kernel_2<<<block_num, num_threads, shared_mem_size>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num, d_data, d_offsets, d_col_index, d_max_nzr, d_x, hll_matrix->M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(z, d_y, hll_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_col_index);
    cudaFree(d_max_nzr);
    cudaFree(d_offsets);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 1000.0;
}