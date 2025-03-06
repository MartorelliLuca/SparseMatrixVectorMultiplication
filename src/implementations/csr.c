#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../headers/csr_headers.h"
#include "../data_structures/csr_matix.h"

// Function to read matrix stored in csr format
csr_matrix *read_CSR_matrix(char *filename)
{
    csr_matrix *matrix = (csr_matrix *)malloc(sizeof(csr_matrix));
    if (matrix == NULL)
    {
        printf("Error in malloc for csr_matrix!\n");
        exit(-1);
    }

    return matrix;
}

// Function to print matrix
void print_csr_matrix(csr_matrix *matrix) {}

// Function to destroy matrix
void destroy_csr_matrix(csr_matrix *matrix) {}
