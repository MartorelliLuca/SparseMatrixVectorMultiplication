#ifndef MATRIX_FORMAT_H
#define MATRIX_FORMAT_H

#include "../utils_header/mmio.h"

typedef struct
{
    char name[256];                   // Matrix Name
    int *rows;                        // Row indices
    int *columns;                     // Columns indices
    double *values;                   // Non zeroes values
    int M;                            // Rows
    int N;                            // Columns
    int number_of_non_zeoroes_values; // Number of nonzeroes values
    MM_typecode matrix_typecode;      // Matrix typecode

} matrix_format;

#endif