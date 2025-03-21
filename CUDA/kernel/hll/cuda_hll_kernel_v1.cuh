#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

__global__ void hll_kernel_v1(int *offsets, int *col_index, double *data, double *x, double *y, int M)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M)
    {
        double sum = 0.0;
        int start = offsets[row];
        int end = offsets[row + 1];

        for (int idx = start; idx < end; idx++)
        {
            int col = col_index[idx];
            sum += data[idx] * x[col];
        }
        y[row] = sum;
    }
}