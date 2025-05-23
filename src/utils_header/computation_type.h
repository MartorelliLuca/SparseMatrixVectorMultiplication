#ifndef COMPUTATION_TIME_H
#define COMPUTATION_TIME_H

typedef enum
{
    SERIAL_CSR,
    SERIAL_HHL,
    PARALLEL_OPEN_MP_CSR,
    PARALLEL_OPEN_MP_HLL,
    CUDA_HLL_KERNEL_1,
    CUDA_HLL_KERNEL_2,
    CUDA_HLL_KERNEL_3,
    CUDA_HLL_KERNEL_4,
    CUDA_CSR_KERNEL_1,
    CUDA_CSR_KERNEL_2,
    CUDA_CSR_KERNEL_3,
    CUDA_CSR_KERNEL_4
} computation_time;

#endif