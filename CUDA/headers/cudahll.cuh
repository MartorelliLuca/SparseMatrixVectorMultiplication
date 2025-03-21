#ifndef CUDAHLL_H
#define CUDAHLL_H

#include "../../src/data_structures/hll_matrix.h"
#include "../../src/data_structures/performance.h"

double prepare_kernel_v1(HLL_matrix *hll_matrix, double *x, double *y);

#endif