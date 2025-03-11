#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

typedef struct
{
    char name[256];   // Matrix Name
    int is_symmetric; // Symmetric Matrix
    int M;            // Number of rows
    int N;            // Number of columns
    int *IRP;         // Pointer to the start of each row in JA and AS
    int *JA;          // Column indices
    int NZ;           // Number of non-zero values
    double *AS;       // Nonzero values
} CSR_matrix;

#endif
