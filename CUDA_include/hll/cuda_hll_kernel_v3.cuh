#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/hll_matrix.h"

/*
******************************************************
*                                                    *
* Third implementation of kernel: shared memory      *
*                                                    *
******************************************************
*/

__global__ void cuda_hll_kernel_v3(double *y, int hack_size, int hacks_num, double *data, int *offsets, int *col_index, int *max_nzr, double *x, int N)
{

    extern __shared__ double shared_x[];

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadIdx.x < N)
        shared_x[threadIdx.x] = x[threadIdx.x];

    if (index < N)
    {
        int hack = index / hack_size;
        int row_start = (index % hack_size) * max_nzr[hack] + offsets[hack];
        int row_end = row_start + max_nzr[hack];
        double sum = 0.0;

        for (int j = row_start; j < row_end; j++)
        {
            if (col_index[j] < 32)
            {
                sum += data[j] * shared_x[col_index[j]];
            }
            else
            {
                sum += data[j] * x[col_index[j]];
            }
        }

        y[index] = sum;
    }
}