#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/hll_matrix.h"

/*
******************************************************
*                                                    *
* Fourth implementation of kernel:                   *
*                                                    *
******************************************************
*/

__global__ void cuda_hll_kernel_v4(double *y, int hack_size, int hacks_num, double *data, int *offsets, int *col_index, int *max_nzr, double *x, int n)
{

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id >> 5; // thread_id / 32
    int lane = thread_id & 31;    // thread_id % 32
    int row = warp_id;

    if (row < n)
    {
        int hack = row / hack_size;
        int row_start = (row % hack_size) * max_nzr[hack] + offsets[hack];
        int row_end = row_start + max_nzr[hack];

        double sum = 0.0;
        for (int element = row_start + lane; element < row_end; element += warpSize)
        {
            sum += data[element] * x[col_index[element]];
        }

        // Warp-wide reduction using shuffle instructions

        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // The first thread in the warp writes the final result
        if (lane == 0)
        {
            y[row] = sum;
        }
    }
}