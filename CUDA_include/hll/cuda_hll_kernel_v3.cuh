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
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    // Caricamento in memoria condivisa
    if (tid < N)
        shared_x[tid] = x[tid];
    __syncthreads();

    if (global_idx < N)
    {
        int hack = global_idx / hack_size;
        int local_offset = (global_idx % hack_size) * max_nzr[hack] + offsets[hack];
        double acc = 0.0;

        for (int j = 0; j < max_nzr[hack]; j++)
        {
            int col = col_index[local_offset + j];
            double value = data[local_offset + j];

            acc += value * (col < 32 ? shared_x[col] : x[col]);
        }

        y[global_idx] = acc;
    }
}