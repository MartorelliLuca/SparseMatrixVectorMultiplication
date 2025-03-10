#ifndef OPERATION_H
#define OPERATION_H

#include "../data_structures/csr_matrix.h"
#include "../data_structures/hll_matrix.h"

void matvec(double **A, double *x, double *y, int M, int N);
void matvec_csr(CSR_matrix *csr_matrix, double *x, double *y);
void matvec_hll(HLL_matrix *ellpack_matrix, double *x, double *y);
int get_real_non_zero_values_count(CSR_matrix *matrix, int block_size, int is_symmetric);

#endif