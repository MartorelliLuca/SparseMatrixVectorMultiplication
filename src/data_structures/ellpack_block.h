#ifndef ELLPACK_BLOCK_H
#define ELLPACK_BLOCK_H

typedef struct
{
    int MAXNZ;   // Max number of nonzero values for every raws;
    int **JA;    // 2D array of column's indices;
    double **AS; // 2D array of coefficients
} ELLPACKBlock;

#endif