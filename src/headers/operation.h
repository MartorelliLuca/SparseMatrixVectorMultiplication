#ifndef OPERATION_H
#define OPERATION_H

#include "../data_structures/csr_matix.h"
#include "../data_structures/hll_matrix.h"

void matvec(double **A, double *x, double *y, int M, int N);
void matvec_csr(CSR_matrix *csr_matrix, double *x, double *y);
void matvec_ellpack(HLL_matrix *ellpack_matrix, double *x, double *y);
int csr_matrix_vector_product_blocked(CSR_matrix *matrix, double *x, double *y, int block_size);

#endif