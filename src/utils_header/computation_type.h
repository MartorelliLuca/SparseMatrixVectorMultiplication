#ifndef COMPUTATION_TIME_H
#define COMPUTATION_TIME_H

typedef enum
{
    SERIAL_CSR,
    SERIAL_HHL,
    PARALLEL_OPEN_MP_CSR,
    PARALLEL_OPEN_MP_HLL,
    CUDA_CSR,
    CUDA_HLL
} computation_time;

#endif