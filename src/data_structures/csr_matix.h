#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

typedef struct
{
    int M;    // Number of rows
    int N;    // Number of columns
    int *IRP; // Pointer to the start of each row in JA and AS
    int *JA;  // Column indices
    int *AS;  // Nonzero values
} csr_matrix;

#endif
