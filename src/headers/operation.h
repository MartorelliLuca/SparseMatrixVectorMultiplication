#ifndef OPERATION_H
#define OPERATION_H

#include "../data_structures/csr_matix.h"
#include "../data_structures/ellpack_matrix.h"

void matvec_csr(CSR_matrix *csr_matrix, double *x, double *y);
void matvec_ellpack(ELLPACK_matrix *ellpack_matrix, double *x, double *y);

#endif