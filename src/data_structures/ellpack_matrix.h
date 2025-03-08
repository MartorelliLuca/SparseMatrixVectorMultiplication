#ifndef ELLPACK_MATRIX_H
#define ELLPACK_MATRIX_H

typedef struct
{
    int M;                // Number of rows
    int N;                // Number of columns
    int MAXNZ;            // Max number of nonzero values for every raws;
    int *row_indices;     // Raw indices
    int *columns_indices; // Columns indices
    int **JA;             // 2D array of column's indices;
    double **AS;          // 2D array of coefficients

} ELLPACK_matrix;

#endif