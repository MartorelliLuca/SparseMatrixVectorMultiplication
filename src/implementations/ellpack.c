#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../data_structures/ellpack_matrix.h"
#include "../headers/ellpack_headers.h"

// Function to read ellpack_matrix stored in csr format
ELLPACK_matrix *read_ELLPACK_matrix(char *filename)
{
    ELLPACK_matrix *ellpack_matrix = (ELLPACK_matrix *)malloc(sizeof(ELLPACK_matrix));
    if (ellpack_matrix == NULL)
    {
        printf("Error in malloc for ellpack ellpack_matrix!\n");
        exit(-1);
    }
    return ellpack_matrix;
}

// Function to print ellpack_matrix
void print_ELLPACK_matrix(ELLPACK_matrix *ellpack_matrix) {}

// Function to destroy ellpack_matrix
void destroy_ELLPACK_matrix(ELLPACK_matrix *ellpack_matrix) {}
