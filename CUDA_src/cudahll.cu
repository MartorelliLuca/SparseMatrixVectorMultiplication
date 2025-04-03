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

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

void print_error(cudaError_t *error, int kernel_index)
{
    if (*error != cudaSuccess)
    {
        // printf("Error occour in invoke kernel %d\nError: %s\n", kernel_index, cudaGetErrorString(*error));
        exit(EXIT_FAILURE);
    }
}

inline void check_cuda(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        // fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n", cudaGetErrorString(err), file, line);
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

    // Memory allocation
    CHECK_CUDA(cudaMalloc(&d_data, hll_matrix->data_num * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_col_index, hll_matrix->data_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_max_nzr, hll_matrix->hacks_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_offsets, hll_matrix->offsets_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x, hll_matrix->N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, hll_matrix->M * sizeof(double)));

    // Copying data
    CHECK_CUDA(cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice));

    // Creating and starting events
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Kernel Execution
    cuda_hll_kernel_v1<<<grid_dim, num_threads>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num,
                                                  d_data, d_offsets, d_col_index, d_max_nzr, d_x, hll_matrix->M);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Time calculation
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy of the result
    CHECK_CUDA(cudaMemcpy(z, d_y, hll_matrix->M * sizeof(double), cudaMemcpyDeviceToHost));

    // Memory Deallocation
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_col_index));
    CHECK_CUDA(cudaFree(d_max_nzr));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    // Destruction of events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds / 1000.0;
}

float invoke_kernel_2(HLL_matrix *hll_matrix, double *x, double *z, int num_threads)
{
    dim3 block_dim(WARP_SIZE, num_threads);
    dim3 grid_dim((hll_matrix->N + block_dim.y - 1) / block_dim.y);

    double *d_data, *d_x, *d_y;
    int *d_col_index, *d_max_nzr, *d_offsets;
    cudaEvent_t start, stop;

    // Memory allocation
    CHECK_CUDA(cudaMalloc(&d_data, hll_matrix->data_num * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_col_index, hll_matrix->data_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_max_nzr, hll_matrix->hacks_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_offsets, hll_matrix->offsets_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x, hll_matrix->N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, hll_matrix->N * sizeof(double)));

    // Copying data
    CHECK_CUDA(cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice));

    // Creating and starting events
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Kernel Execution
    cuda_hll_kernel_v2<<<grid_dim, num_threads>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num, d_data, d_offsets,
                                                  d_col_index, d_max_nzr, d_x, hll_matrix->M);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Time calculation
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy of the result
    CHECK_CUDA(cudaMemcpy(z, d_y, hll_matrix->M * sizeof(double), cudaMemcpyDeviceToHost));

    // Memory Deallocation
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_col_index));
    CHECK_CUDA(cudaFree(d_max_nzr));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    // Destruction of events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds / 1000.0;
}

float invoke_kernel_3(HLL_matrix *hll_matrix, double *x, double *z, int num_threads)
{
    dim3 block_dim(WARP_SIZE, num_threads);
    dim3 grid_dim((hll_matrix->N + block_dim.y - 1) / block_dim.y);

    double *d_data, *d_x, *d_y;
    int *d_col_index, *d_max_nzr, *d_offsets;
    cudaEvent_t start, stop;

    // Memory allocation
    CHECK_CUDA(cudaMalloc(&d_data, hll_matrix->data_num * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_col_index, hll_matrix->data_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_max_nzr, hll_matrix->hacks_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_offsets, hll_matrix->offsets_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x, hll_matrix->N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, hll_matrix->N * sizeof(double)));

    // Copying data
    CHECK_CUDA(cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice));

    // Creating and starting events
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Kernel Execution
    int shared_mem_size = 1024 * sizeof(double);
    cuda_hll_kernel_v3<<<grid_dim, num_threads, shared_mem_size>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num,
                                                                   d_data, d_offsets, d_col_index, d_max_nzr, d_x,
                                                                   hll_matrix->M);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Time calculation
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy of the result
    CHECK_CUDA(cudaMemcpy(z, d_y, hll_matrix->N * sizeof(double), cudaMemcpyDeviceToHost));

    // Memory Deallocation
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_col_index));
    CHECK_CUDA(cudaFree(d_max_nzr));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    // Destruction of events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds / 1000.0;
}

float invoke_kernel_4(HLL_matrix *hll_matrix, double *x, double *z, int num_threads)
{
    int block_num = (hll_matrix->N * 32) / num_threads;

    double *d_data, *d_x, *d_y;
    int *d_col_index, *d_max_nzr, *d_offsets;
    cudaEvent_t start, stop;

    // Memory allocation
    CHECK_CUDA(cudaMalloc(&d_data, hll_matrix->data_num * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_col_index, hll_matrix->data_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_max_nzr, hll_matrix->hacks_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_offsets, hll_matrix->offsets_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x, hll_matrix->N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, hll_matrix->N * sizeof(double)));

    // Copying data
    CHECK_CUDA(cudaMemcpy(d_data, hll_matrix->data, hll_matrix->data_num * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_index, hll_matrix->col_index, hll_matrix->data_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_max_nzr, hll_matrix->max_nzr, hll_matrix->hacks_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, hll_matrix->offsets, hll_matrix->offsets_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, hll_matrix->N * sizeof(double), cudaMemcpyHostToDevice));

    // Creating and starting events
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Kernel Execution
    cuda_hll_kernel_v4<<<block_num, num_threads>>>(d_y, hll_matrix->hack_size, hll_matrix->hacks_num,
                                                   d_data, d_offsets, d_col_index, d_max_nzr, d_x,
                                                   hll_matrix->M);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Time calculation
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy of the result
    CHECK_CUDA(cudaMemcpy(z, d_y, hll_matrix->N * sizeof(double), cudaMemcpyDeviceToHost));

    // Memory Deallocation
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_col_index));
    CHECK_CUDA(cudaFree(d_max_nzr));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    // Destruction of events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds / 1000.0;
}
