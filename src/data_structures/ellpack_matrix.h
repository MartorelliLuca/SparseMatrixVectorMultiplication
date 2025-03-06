#ifndef ELLPACK_MATRIX_H
#define ELLPACK_MATRIX_H

typedef struct
{
    int M;     // Number of rows
    int N;     // Number of columns
    int MAXNZ; // Max number of nonzero values for every raws;
    int **JA;  // 2D array of column's indexes;
    int **AS;  // 2D array of coefficients

} ellpack_matrix;

#endif