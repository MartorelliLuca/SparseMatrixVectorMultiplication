#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../data_structures/ellpack_matrix.h"
#include "../headers/ellpack_headers.h"

// Function to read matrix stored in csr format
ellpack_matrix *read_ellpack_matrix(char *filename)
{
    ellpack_matrix *matrix = (ellpack_matrix *)malloc(sizeof(ellpack_matrix));
    if (matrix == NULL)
    {
        printf("Error in malloc for ellpack matrix!\n");
        exit(-1);
    }
    return matrix;
}

// Function to print matrix
void print_ellpack_matrix(ellpack_matrix *matrix) {}

// Function to destroy matrix
void destroy_ellpack_matrix(ellpack_matrix *matrix) {}
