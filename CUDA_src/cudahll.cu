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
#include "../CUDA_include/hll/cuda_hll_kernel_v3.cuh"
#include "../CUDA_include/hll/cuda_hll_kernel_v4.cuh"

#define WARP_SIZE 32

void print_error(cudaError_t *error, int kernel_index)
{
    if (*error != cudaSuccess)
    {
        printf("Error occour in invoke kernel %d\nError: %s\n", kernel_index, cudaGetErrorString(*error));
        exit(EXIT_FAILURE);
    }
}

float invoke_kernel_1(HLL_matrix *hll_matrix, double *x, double *z, int num_threads)
{
    dim3 block_dim(WARP_SIZE, num_threads);
    dim3 grid_dim((hll_matrix->N + block_dim.y - 1) / block_dim.y);

    double *d_data, *d_x, *d_y;
    int *d_col_index, *d_max_nzr, *d_offsets;
    cudaEvent_t start, stop;
    cudaError_t error;

    error = cudaMalloc(&d_data, hll_matrix->data_num * sizeof(double));
    print_error(&error, 1);

    error = cudaMalloc(&d_col_index, hll_matrix->data_num * sizeof(int));
    print_error(&error, 1);

    error = cudaMalloc(&d_max_nzr, hll_matrix->hacks_num * sizeof(int));
    print_error(&error, 1);

    error = cudaMalloc(&d_offsets, hll_matrix->offsets_num * sizeof(int));
    print_error(&error, 1);

    error = cudaMalloc(&d_x, hll_matrix->N * sizeof(double));
    print_error(&error, 1);

    error = cudaMalloc(&d_y, hll_matrix->M * sizeof(double));
    print_error(&error, 1);

    error = cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice);
    print_error(&error, 1);

    error = cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 1);

    error = cudaMemcpy(d_max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 1);

    error = cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 1);

    error = cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice);
    print_error(&error, 1);

    error = cudaEventCreate(&start);
    print_error(&error, 1);

    error = cudaEventCreate(&stop);
    print_error(&error, 1);

    error = cudaEventRecord(start);
    print_error(&error, 1);

    cuda_hll_kernel_v1<<<grid_dim, num_threads>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num, d_data, d_offsets, d_col_index,
                                                  d_max_nzr, d_x, hll_matrix->M);

    error = cudaEventRecord(stop);
    print_error(&error, 1);

    error = cudaEventSynchronize(stop);
    print_error(&error, 1);

    float milliseconds = 0;
    error = cudaEventElapsedTime(&milliseconds, start, stop);
    print_error(&error, 1);

    error = cudaMemcpy(z, d_y, hll_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);
    print_error(&error, 1);

    error = cudaFree(d_data);
    print_error(&error, 1);

    error = cudaFree(d_col_index);
    print_error(&error, 1);

    error = cudaFree(d_max_nzr);
    print_error(&error, 1);

    error = cudaFree(d_offsets);
    print_error(&error, 1);

    error = cudaFree(d_x);
    print_error(&error, 1);

    error = cudaFree(d_y);
    print_error(&error, 1);

    error = cudaEventDestroy(start);
    print_error(&error, 1);

    error = cudaEventDestroy(stop);
    print_error(&error, 1);

    return milliseconds / 1000.0;
}

float invoke_kernel_2(HLL_matrix *hll_matrix, double *x, double *z, int num_threads)
{
    dim3 block_dim(WARP_SIZE, num_threads);
    dim3 grid_dim((hll_matrix->N + block_dim.y - 1) / block_dim.y);

    double *d_data, *d_x, *d_y;
    int *d_col_index, *d_max_nzr, *d_offsets;
    cudaEvent_t start, stop;
    cudaError_t error;

    error = cudaMalloc(&d_data, hll_matrix->data_num * sizeof(double));
    print_error(&error, 2);

    error = cudaMalloc(&d_col_index, hll_matrix->data_num * sizeof(int));
    print_error(&error, 2);

    error = cudaMalloc(&d_max_nzr, hll_matrix->hacks_num * sizeof(int));
    print_error(&error, 2);

    error = cudaMalloc(&d_offsets, hll_matrix->offsets_num * sizeof(int));
    print_error(&error, 2);

    error = cudaMalloc(&d_x, hll_matrix->N * sizeof(double));
    print_error(&error, 2);

    error = cudaMalloc(&d_y, hll_matrix->N * sizeof(double));
    print_error(&error, 2);

    error = cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice);
    print_error(&error, 2);

    error = cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 2);

    error = cudaMemcpy(d_max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 2);

    error = cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 2);

    error = cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice);
    print_error(&error, 2);

    error = cudaEventCreate(&start);
    print_error(&error, 2);

    error = cudaEventCreate(&stop);
    print_error(&error, 2);

    error = cudaEventRecord(start);
    print_error(&error, 2);

    cuda_hll_kernel_v2<<<grid_dim, num_threads>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num, d_data, d_offsets,
                                                  d_col_index, d_max_nzr, d_x, hll_matrix->M);

    error = cudaEventRecord(stop);
    print_error(&error, 2);

    error = cudaEventSynchronize(stop);
    print_error(&error, 2);

    float milliseconds = 0;
    error = cudaEventElapsedTime(&milliseconds, start, stop);
    print_error(&error, 2);

    error = cudaMemcpy(z, d_y, hll_matrix->M * sizeof(double), cudaMemcpyDeviceToHost);
    print_error(&error, 2);

    error = cudaFree(d_data);
    print_error(&error, 2);

    error = cudaFree(d_col_index);
    print_error(&error, 2);

    error = cudaFree(d_max_nzr);
    print_error(&error, 2);

    error = cudaFree(d_offsets);
    print_error(&error, 2);

    error = cudaFree(d_x);
    print_error(&error, 2);

    error = cudaFree(d_y);
    print_error(&error, 2);

    error = cudaEventDestroy(start);
    print_error(&error, 2);

    error = cudaEventDestroy(stop);
    print_error(&error, 2);

    return milliseconds / 1000.0;
}

float invoke_kernel_3(HLL_matrix *hll_matrix, double *x, double *z, int num_threads)
{
    dim3 block_dim(WARP_SIZE, num_threads);
    dim3 grid_dim((hll_matrix->N + block_dim.y - 1) / block_dim.y);

    double *d_data, *d_x, *d_y;
    int *d_col_index, *d_max_nzr, *d_offsets;
    cudaEvent_t start, stop;
    cudaError_t error;

    error = cudaMalloc(&d_data, hll_matrix->data_num * sizeof(double));
    print_error(&error, 3);

    error = cudaMalloc(&d_col_index, hll_matrix->data_num * sizeof(int));
    print_error(&error, 3);

    error = cudaMalloc(&d_max_nzr, hll_matrix->hacks_num * sizeof(int));
    print_error(&error, 3);

    error = cudaMalloc(&d_offsets, hll_matrix->offsets_num * sizeof(int));
    print_error(&error, 3);

    error = cudaMalloc(&d_x, hll_matrix->N * sizeof(double));
    print_error(&error, 3);

    error = cudaMalloc(&d_y, hll_matrix->N * sizeof(double));
    print_error(&error, 3);

    error = cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice);
    print_error(&error, 3);

    error = cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 3);

    error = cudaMemcpy(d_max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 3);

    error = cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 3);

    error = cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice);
    print_error(&error, 3);

    error = cudaEventCreate(&start);
    print_error(&error, 3);

    error = cudaEventCreate(&stop);
    print_error(&error, 3);

    error = cudaEventRecord(start);
    print_error(&error, 3);

    int shared_mem_size = 1024 * sizeof(double);
    cuda_hll_kernel_v3<<<grid_dim, num_threads, shared_mem_size>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num,
                                                                   d_data, d_offsets, d_col_index, d_max_nzr, d_x,
                                                                   hll_matrix->M);

    error = cudaEventRecord(stop);
    print_error(&error, 3);

    error = cudaEventSynchronize(stop);
    print_error(&error, 3);

    float milliseconds = 0;
    error = cudaEventElapsedTime(&milliseconds, start, stop);
    print_error(&error, 3);

    error = cudaMemcpy(z, d_y, hll_matrix->N * sizeof(double), cudaMemcpyDeviceToHost);
    print_error(&error, 3);

    error = cudaFree(d_data);
    print_error(&error, 3);

    error = cudaFree(d_col_index);
    print_error(&error, 3);

    error = cudaFree(d_max_nzr);
    print_error(&error, 3);

    error = cudaFree(d_offsets);
    print_error(&error, 3);

    error = cudaFree(d_x);
    print_error(&error, 3);

    error = cudaFree(d_y);
    print_error(&error, 3);

    error = cudaEventDestroy(start);
    print_error(&error, 3);

    error = cudaEventDestroy(stop);
    print_error(&error, 3);

    return milliseconds / 1000.0;
}

float invoke_kernel_4(HLL_matrix *hll_matrix, double *x, double *z, int num_threads)
{
    int block_num = (hll_matrix->N * 32) / num_threads;

    double *d_data, *d_x, *d_y;
    int *d_col_index, *d_max_nzr, *d_offsets;
    cudaEvent_t start, stop;
    cudaError_t error;

    error = cudaMalloc(&d_data, hll_matrix->data_num * sizeof(double));
    print_error(&error, 4);

    error = cudaMalloc(&d_col_index, hll_matrix->data_num * sizeof(int));
    print_error(&error, 4);

    error = cudaMalloc(&d_max_nzr, hll_matrix->hacks_num * sizeof(int));
    print_error(&error, 4);

    error = cudaMalloc(&d_offsets, hll_matrix->offsets_num * sizeof(int));
    print_error(&error, 4);

    error = cudaMalloc(&d_x, hll_matrix->N * sizeof(double));
    print_error(&error, 4);

    error = cudaMalloc(&d_y, hll_matrix->N * sizeof(double));
    print_error(&error, 4);

    error = cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice);
    print_error(&error, 4);

    error = cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 4);

    error = cudaMemcpy(d_max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 4);

    error = cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice);
    print_error(&error, 4);

    error = cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice);
    print_error(&error, 4);

    error = cudaEventCreate(&start);
    print_error(&error, 4);

    error = cudaEventCreate(&stop);
    print_error(&error, 4);

    error = cudaEventRecord(start);
    print_error(&error, 4);

    cuda_hll_kernel_v4<<<block_num, num_threads>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num,
                                                   d_data, d_offsets, d_col_index, d_max_nzr, d_x,
                                                   hll_matrix->M);

    error = cudaEventRecord(stop);
    print_error(&error, 4);

    error = cudaEventSynchronize(stop);
    print_error(&error, 4);

    float milliseconds = 0;
    error = cudaEventElapsedTime(&milliseconds, start, stop);
    print_error(&error, 4);

    error = cudaMemcpy(z, d_y, hll_matrix->N * sizeof(double), cudaMemcpyDeviceToHost);
    print_error(&error, 4);

    error = cudaFree(d_data);
    print_error(&error, 4);

    error = cudaFree(d_col_index);
    print_error(&error, 4);

    error = cudaFree(d_max_nzr);
    print_error(&error, 4);

    error = cudaFree(d_offsets);
    print_error(&error, 4);

    error = cudaFree(d_x);
    print_error(&error, 4);

    error = cudaFree(d_y);
    print_error(&error, 4);

    error = cudaEventDestroy(start);
    print_error(&error, 4);

    error = cudaEventDestroy(stop);
    print_error(&error, 4);

    return milliseconds / 1000.0;
}