#ifndef HLL_MATRIX_H
#define HLL_MATRIX_H

#define HACKSIZE 32

#include "ellpack_block.h"

typedef struct
{
    char name[256]; // Matrix Name
    int M;          // Number of rows
    int N;          // Number of columns
    int *offest;
    int num_offset;
    int *columns;
    double *values;
    int num_values;
    int num_hack;
    int hack_size;
    int *max_non_zeroes;
} HLL_matrix;

#endif