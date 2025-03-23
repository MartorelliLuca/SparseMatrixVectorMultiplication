#ifndef INVOKE_HLL_KENREL_H
#define INVOKE_HLL_KENREL_H

#include "../data_structures/hll_matrix.h"
#include "../data_structures/performance.h"

void invoke_cuda_hll_kernels(HLL_matrix *hll_matrix, double *x, double *z, double *effective_results, struct performance *head, struct performance *tail, struct performance *node);

#endif