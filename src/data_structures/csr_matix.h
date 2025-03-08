#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

typedef struct
{
    int M;                 // Number of rows
    int N;                 // Number of columns
    int *IRP;              // Pointer to the start of each row in JA and AS
    int *JA;               // Column indices
    int NZ;                // Number of non-zero values
    int *row_indices;      // Raw indices
    int *columns_indices;  // Columns indices
    double *readed_values; // Readed values
    double *AS;            // Nonzero values
} CSR_matrix;

#endif
