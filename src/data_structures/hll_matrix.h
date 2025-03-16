#ifndef HLL_MATRIX_H
#define HLL_MATRIX_H

#include <stdlib.h>
#include "../headers/matrix_format.h"

#define HACKSIZE 32

typedef struct
{
    char name[256];
    int M;
    int N;
    int *offsets;
    int offsets_num;
    int *col_index;
    double *data;
    int data_num;
    int hacks_num;
    int hack_size;
    int *max_nzr;
} HLL_matrix;

#endif