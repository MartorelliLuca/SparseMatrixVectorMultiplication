#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/hll_matrix.h"

__global__ void cuda_kernel_2(double *__restrict__ res, int hack_size, int hacks_num, double *__restrict__ data, int *__restrict__ offsets, int *__restrict__ col_index, int *__restrict__ max_nzr, double *__restrict__ x, int M)
{
    extern __shared__ double x_shared[];
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

        for (int j = threadIdx.x; j < max_nzr_h; j += blockDim.x)
        {
            int col = col_index[offset + j];
            double val = data[offset + j];
            sum += val * x[col];
        }

        atomicAdd(&res[r], sum); // Uso di atomicAdd per evitare race conditions
    }
}