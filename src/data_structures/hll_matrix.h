#ifndef HLL_MATRIX_H
#define HLL_MATRIX_H

#define HACKSIZE 32

#include "ellpack_block.h"

typedef struct
{
    char name[256];        // Matrix Name
    ELLPACK_block *blocks; // Blocks Array
    int number_of_blocks;  // Blocks number
} HLL_matrix;

#endif