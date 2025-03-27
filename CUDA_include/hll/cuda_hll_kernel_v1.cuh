#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/hll_matrix.h"

/*
*****************************************************
*                                                   *
* First implementation of kernel: 1 thread for hack *
*                                                   *
*****************************************************
*/

__global__ void cuda_hll_kernel_v1(double *y, int hack_size, int hacks_num, double *data, int *offsets, int *col_index, int *max_nzr, double *x, int M)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= hacks_num)
        return;

    int start_row = index * hack_size;
    int end_row = min(start_row + hack_size, M);
    int max_nzr_h = max_nzr[index];
    int base_offset = offsets[index];

    for (int row_index = start_row; row_index < end_row; ++row_index)
    {
        double sum = 0.0;
        int row_offset = base_offset + (row_index - start_row) * max_nzr_h;

        for (int j = 0; j < max_nzr_h; ++j)
        {
            sum += data[row_offset + j] * x[col_index[row_offset + j]];
        }

        y[row_index] = sum;
    }
}
