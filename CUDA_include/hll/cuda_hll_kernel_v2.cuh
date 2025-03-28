#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/hll_matrix.h"

/*
******************************************************
*                                                    *
* Second implementation of kernel: 1 thread for rows *
*                                                    *
******************************************************
*/

__global__ void cuda_hll_kernel_v2(double *y, int hack_size, int hacks_num, double *data, int *offsets, int *col_index, int *max_nzr, double *x, int N)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N)
        return;

    int hack = index / hack_size;
    int local_row = index % hack_size;

    int row_start = offsets[hack] + local_row * max_nzr[hack];
    int row_end = row_start + max_nzr[hack];

    double sum = 0.0;
    for (int j = row_start; j < row_end; ++j)
    {
        sum += data[j] * x[col_index[j]];
    }

    y[index] = sum;
}