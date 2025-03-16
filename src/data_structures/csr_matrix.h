#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

typedef struct
{
    char name[256];
    int *JA;
    int *IRP;
    double *AS;
    int M;
    int N;
    int non_zero_values;
} CSR_matrix;

#endif
