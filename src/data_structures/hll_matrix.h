#ifndef HLL_MATRIX_H
#define HLL_MATRIX_H

#include "ellpack_block.h"

typedef struct
{
    int hack_size;        // Hack Size Parameter
    int num_blocks;       // Num Blocks
    int M;                // Number of rows
    int N;                // Number of columns
    ELLPACKBlock *blocks; // ELLPACK Blocks

} HLL_matrix;

#endif