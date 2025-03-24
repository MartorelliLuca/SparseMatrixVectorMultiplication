#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/hll_matrix.h"

__global__ void cuda_kernel_0(double *res, int hack_size, int hacks_num, double *data, int *offsets, int *col_index, int *max_nzr, double *x, int N)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    if (h < hacks_num)
    {
        int rows;
        if (0 == hacks_num - 1 && N % hack_size != 0)
        {
            rows = N % hack_size;
        }
        else
        {
            rows = hack_size;
        }

        int current_rows;
        if (h == hacks_num - 1 && N % hack_size != 0)
        {
            current_rows = N % hack_size;
        }
        else
        {
            current_rows = hack_size;
        }

        for (int r = 0; r < current_rows; ++r)
        {
            double sum = 0.0;
            for (int j = 0; j < max_nzr[h]; ++j)
            {
                int k = offsets[h] + r * max_nzr[h] + j;
                sum += data[k] * x[col_index[k]];
            }
            res[rows * h + r] = sum;
        }
    }
}
