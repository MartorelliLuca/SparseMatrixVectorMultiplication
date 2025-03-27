#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/hll_matrix.h"

__global__ void cuda_hll_kernel_v3(double *y, int hack_size, int hacks_num, double *data, int *offsets, int *col_index, int *max_nzr, double *v, int N)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    int hack = index / hack_size;
    int local_row = index % hack_size;

    __shared__ int shared_max_nzr;
    __shared__ int shared_offset;

    if (threadIdx.x == 0)
    {
        shared_max_nzr = max_nzr[hack];
        shared_offset = offsets[hack];
    }

    __syncthreads();

    int row_start = shared_offset + local_row * shared_max_nzr;
    int row_end = row_start + shared_max_nzr;

    extern __shared__ double shared_data[];
    int *shared_col_index = (int *)&shared_data[shared_max_nzr];

    for (int j = threadIdx.x; j < shared_max_nzr; j += blockDim.x)
    {
        shared_col_index[j] = col_index[row_start + j];
        shared_data[j] = data[row_start + j];
    }

    __syncthreads();

    double sum = 0.0;
    for (int j = 0; j < shared_max_nzr; ++j)
    {
        sum += shared_data[j] * v[shared_col_index[j]];
    }

    y[index] = sum;
}