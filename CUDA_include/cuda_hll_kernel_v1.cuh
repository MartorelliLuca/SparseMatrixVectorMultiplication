#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../src/data_structures/hll_matrix.h"

__global__ void hll_matvec_kernel(HLL_matrix d_A, double *d_x, double *d_y)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < d_A.M)
    {
        double sum = 0.0;

        int hack_id = row / d_A.hack_size;
        int start_offset = d_A.offsets[hack_id];
        int end_offset = (hack_id < d_A.hacks_num - 1) ? d_A.offsets[hack_id + 1] : d_A.data_num;
        int max_nonzeros = d_A.max_nzr[hack_id];

        int local_row = row % d_A.hack_size;

        for (int i = 0; i < max_nonzeros; i++)
        {
            int data_index = start_offset + local_row * max_nonzeros + i;
            if (data_index < end_offset)
            {
                int col = d_A.col_index[data_index];
                double val = d_A.data[data_index];
                sum += val * d_x[col];
            }
        }

        d_y[row] = sum;
    }
}