#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../data_structures/hll_matrix.h"
#include "../headers/hll_headers.h"

// Function to read ellpack_matrix stored in csr format
HLL_matrix *read_ELLPACK_matrix(char *filename)
{
    HLL_matrix *ellpack_matrix = (HLL_matrix *)malloc(sizeof(HLL_matrix));
    if (ellpack_matrix == NULL)
    {
        printf("Error in malloc for ellpack ellpack_matrix!\n");
        exit(-1);
    }
    return ellpack_matrix;
}

// Function to print ellpack_matrix
void print_ELLPACK_matrix(HLL_matrix *ellpack_matrix) {}

// Function to destroy ellpack_matrix
void destroy_ELLPACK_matrix(HLL_matrix *ellpack_matrix) {}
