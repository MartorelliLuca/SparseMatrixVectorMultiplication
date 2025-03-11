#ifndef HLL_MATRIX_H
#define HLL_MATRIX_H

#define HACKSIZE 32

typedef struct
{
    int M, N;          // Rows and Columns number
    int hack_size;     // Hack dimension
    int num_hacks;     // Hack Number
    int *hack_offsets; // Start indices of hcks
    int *JA;           // Columns indicies
    double *AS;        // Non-zeroes values
} HLL_matrix;

#endif