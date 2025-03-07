#ifndef READER_MATRIX_H
#define READER_MATRIX_H

#include "../data_structures/csr_matix.h"
#include "../data_structures/ellpack_matrix.h"

void convert_to_ellpack(int M, int N, int nz, int *row_indices, int *col_indices, double *values, int **JA, double **AS, int *MAXNZ);
void convert_to_csr(int M, int N, int nz, int *row_indices, int *col_indices, double *values, int **IRP, int **JA, double **AS);
void matvec_csr(int M, int *IRP, int *JA, double *AS, double *x, double *y);
void matvec_csr(int M, int *IRP, int *JA, double *AS, double *x, double *y);

#endif