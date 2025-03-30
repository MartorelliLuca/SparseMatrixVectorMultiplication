#ifndef INVOKE_CSR_KERNEL_H
#define INVOKE_CSR_KERNEL_H

#include "../data_structures/csr_matrix.h"
#include "../data_structures/performance.h"

void invoke_cuda_csr_kernels(CSR_matrix *csr_matrix, double *x, double *z, double *effective_results, struct performance **head, struct performance **tail, struct performance *node);

#endif