#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

__global__ void csr_matvec_warps_per_row(int n, int *d_row, int *d_col, double *d_val, double *d_x, double *d_y, double *d_res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double sum = 0.0;
        for (int j = d_row[i]; j < d_row[i + 1]; j++)
        {
            sum += d_val[j] * d_x[d_col[j]];
        }
        d_y[i] = sum;
    }
    if (i == 0)
    {
        double res = 0.0;
        for (int j = 0; j < n; j++)
        {
            res += d_y[j];
        }
        *d_res = res;
    }
}
