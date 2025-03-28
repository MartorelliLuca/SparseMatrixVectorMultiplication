#ifndef CUDACSR_H
#define CUDACSR_H

#include "../src/headers/csr_headers.h"
#include "../src/data_structures/csr_matrix.h"

#ifdef __cplusplus
extern "C"
{
#endif

    float invoke_kernel_csr_1(CSR_matrix *csr, double *x, double *y, int num_threads_per_block);
    float invoke_kernel_csr_2(CSR_matrix *csr, double *x, double *y, int num_threads_per_block);
    float invoke_kernel_csr_3(CSR_matrix *csr, double *x, double *y, int num_threads_per_block);
    float invoke_kernel_csr_4(CSR_matrix *csr, double *x, double *y, int num_threads_per_block);

#ifdef __cplusplus
}
#endif

#endif
