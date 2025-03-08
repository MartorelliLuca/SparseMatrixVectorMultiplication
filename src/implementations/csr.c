#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../headers/csr_headers.h"
#include "../data_structures/csr_matix.h"

// Function to read matrix stored in csr format
CSR_matrix *read_CSR_matrix(char *filename)
{
    CSR_matrix *csr_matrix = (CSR_matrix *)malloc(sizeof(CSR_matrix));
    if (csr_matrix == NULL)
    {
        printf("Error in malloc for csr_matrix!\n");
        exit(-1);
    }

    return csr_matrix;
}

// Function to destroy matrix
void destroy_CSR_matrix(CSR_matrix *csr_matrix) {}
