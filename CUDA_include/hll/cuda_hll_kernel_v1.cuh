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
*                                                   *
*****************************************************
*/

__global__ void cuda_kernel_1(double *res, int hack_size, int hacks_num, double *data, int *offsets, int *col_index, int *max_nzr, double *x, int M)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hacks_num)
        return;

    int start_row = h * hack_size;
    int end_row = min(start_row + hack_size, M);

    for (int r = start_row; r < end_row; ++r)
    {
        double sum = 0.0;
        int max_nzr_h = max_nzr[h];
        int offset = offsets[h] + (r - start_row) * max_nzr_h;

        for (int j = 0; j < max_nzr_h; ++j)
        {
            int col = col_index[offset + j];
            double val = data[offset + j];
            sum += val * x[col];
        }
        res[r] = sum;
    }
}
