#ifndef ELLPACK_BLOCK_H
#define ELLPACK_BLOCK_H

typedef struct
{
    int max_non_zeroes_per_row; // Max number of nonzero values for every raws;
    int array_size;             // Array size
    int non_zeroes_per_block;   // Non zeroes per block
    int *JA;                    // Array of column's indices;
    double *AS;                 // Array of coefficients
} ELLPACK_block;

#endif