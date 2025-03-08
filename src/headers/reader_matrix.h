#ifndef READER_MATRIX_H
#define READER_MATRIX_H

#include "../data_structures/csr_matix.h"
#include "../data_structures/ellpack_matrix.h"

void convert_to_ellpack(ELLPACK_matrix *ellpack_matrix);
void convert_to_csr(CSR_matrix *CSR_matrix);

void matvec_csr(CSR_matrix *csr_matrix, double *x, double *y);
void matvec_ellpack(ELLPACK_matrix *ellpack_matrix, double *x, double *y);

void print_matrix(CSR_matrix *csr_matrix);

#endif