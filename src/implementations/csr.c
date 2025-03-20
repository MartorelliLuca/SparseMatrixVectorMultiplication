#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <string.h>

#include "../utils_header/initialization.h"
#include "../headers/csr_headers.h"
#include "../headers/matrix_format.h"
#include "../data_structures/csr_matrix.h"
#include "../utils_header/mmio.h"

void read_CSR_matrix(FILE *matrix_file, CSR_matrix *csr_matrix, matrix_format *matrix)
{
    csr_matrix->M = matrix->M;
    csr_matrix->N = matrix->N;
    csr_matrix->non_zero_values = matrix->number_of_non_zeoroes_values;

    csr_matrix->IRP = (int *)malloc((csr_matrix->M + 1) * sizeof(int));
    if (!csr_matrix->IRP)
    {
        printf("Errore in malloc per irp in read csr matrix\n");
        exit(EXIT_FAILURE);
    }

    csr_matrix->JA = (int *)malloc(csr_matrix->non_zero_values * sizeof(int));
    if (!csr_matrix->JA)
    {
        printf("Errore in malloc per ja in read csr matrix!\n");
        exit(EXIT_FAILURE);
    }

    csr_matrix->AS = (double *)malloc(csr_matrix->non_zero_values * sizeof(double));
    if (!csr_matrix->AS)
    {
        printf("Errore in malloc per as in read csr matrix\n");
        exit(EXIT_FAILURE);
    }

    if (matrix->values == NULL)
    {
        printf("matrix->values null\n");
        return;
    }

    if (matrix->columns == NULL)
    {
        printf("matrix->values null\n");
        return;
    }

    memcpy(csr_matrix->AS, matrix->values, matrix->number_of_non_zeoroes_values * sizeof(double));
    memcpy(csr_matrix->JA, matrix->columns, matrix->number_of_non_zeoroes_values * sizeof(int));

    csr_matrix->IRP[0] = 0;

    int prev = matrix->rows[0];
    int k = 1;
    for (int i = 0; i < matrix->number_of_non_zeoroes_values; ++i)
    {
        int curr = matrix->rows[i];
        while (curr - prev > 1)
        {
            csr_matrix->IRP[k++] = i;
            ++prev;
        }
        if (curr - prev == 1)
        {
            prev = curr;
            csr_matrix->IRP[k++] = i;
        }
    }
    while (k <= matrix->M)
    {
        csr_matrix->IRP[k++] = matrix->number_of_non_zeoroes_values;
    }
}

// Function to destroy matrix
void destroy_CSR_matrix(CSR_matrix *csr_matrix)
{
    if (!csr_matrix)
        return;

    free(csr_matrix->IRP);
    free(csr_matrix->JA);
    free(csr_matrix->AS);
    free(csr_matrix);
}
