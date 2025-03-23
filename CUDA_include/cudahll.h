#ifndef CUDAHLL_H
#define CUDAHLL_H

#include "../src/data_structures/hll_matrix.h"
#include "../src/data_structures/performance.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void invoke_kernel_1(HLL_matrix *hll_matrix, double *x, double *z, float *time);

#ifdef __cplusplus
}
#endif

#endif
