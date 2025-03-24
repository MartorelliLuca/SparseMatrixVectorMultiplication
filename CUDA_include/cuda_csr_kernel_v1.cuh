#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../src/data_structures/csr_matrix.h"

__global__ void csr_matvec_kernel(CSR_matrix d_A, double *d_x, double *d_y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < d_A.M)
    {
        double sum = 0.0;
        int start = d_A.IRP[row];
        int end = d_A.IRP[row + 1];
        for (int i = start; i < end; i++)
        {
            int col = d_A.JA[i];
            double val = d_A.AS[i];
            sum += val * d_x[col];
        }
        d_y[row] = sum;
    }
}