#include <ctype.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <bits/time.h>
#include <errno.h>

#include "../data_structures/csr_matix.h"
#include "../data_structures/ellpack_matrix.h"
#include "../headers/reader_matrix.h"

// Function to convert csr_matrix in HLL format
void convert_to_ellpack(ELLPACK_matrix *ellpack_matrix)
{
}

// Function to convert csr_matrix in CSR format
void convert_to_csr(CSR_matrix *csr_matrix)
{
    int *row_positions;
    int row, position;
    // Create array for csr_matrix struct
    csr_matrix->IRP = (int *)malloc(((csr_matrix->M) + 1) * sizeof(int));
    if (csr_matrix->IRP == NULL)
    {
        printf("Error occour in malloc for IRP in csr_matrix conversion!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    csr_matrix->JA = (int *)malloc((csr_matrix->NZ) * sizeof(int));
    if (csr_matrix->JA == NULL)
    {
        printf("Error occour in malloc for JA in csr_matrix convertion!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    csr_matrix->AS = (double *)malloc((csr_matrix->NZ) * sizeof(double));
    if (csr_matrix->AS == NULL)
    {
        printf("Error occour in malloc for AS in csr_matrix convertion!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // Count the non zero values for every raws
    for (int i = 0; i < csr_matrix->NZ; i++)
        csr_matrix->IRP[csr_matrix->row_indices[i] + 1]++;

    // Create the IRP vector
    for (int i = 1; i <= csr_matrix->M; i++)
        csr_matrix->IRP[i] += csr_matrix->IRP[i - 1];

    // Assign to JA and AS the values
    row_positions = (int *)malloc(csr_matrix->M * sizeof(int));
    if (row_positions == NULL)
    {
        printf("Error occour in malloc for row position in convert to csr_matrix format function!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }
    memcpy(row_positions, csr_matrix->IRP, csr_matrix->M * sizeof(int));

    for (int i = 0; i < csr_matrix->NZ; i++)
    {
        row = csr_matrix->row_indices[i];
        position = row_positions[row]++;
        csr_matrix->JA[position] = csr_matrix->columns_indices[i];
        csr_matrix->AS[position] = csr_matrix->readed_values[i];
    }

    // print_matrix(csr_matrix);

    free(row_positions);
}

// First attempt to do matrix-vector dot product in CSR format
void matvec_csr(CSR_matrix *csr_matrix, double *x, double *y)
{
    for (int i = 0; i < csr_matrix->M; i++)
    {
        for (int j = csr_matrix->IRP[i]; j < csr_matrix->IRP[i + 1]; j++)
        {
            y[i] += csr_matrix->AS[j] * x[csr_matrix->JA[j]];
        }
    }
}

// First attempt to do matrix-vector dot produt in ELLPACK format
void matvec_ellpack(ELLPACK_matrix *ellpack_matrix, double *x, double *y)
{
}

void print_matrix(CSR_matrix *csr_matrix)
{
    printf("CSR Matrix");
    printf("M = %d, N = %d, NZ = %d\n", csr_matrix->M, csr_matrix->N, csr_matrix->NZ);

    printf("IRP: ");
    for (int i = 0; i <= csr_matrix->M; i++)
        printf("%d ", csr_matrix->IRP[i]);
    printf("\n");

    printf("JA: ");
    for (int i = 0; i < csr_matrix->NZ; i++)
        printf("%d ", csr_matrix->JA[i]);
    printf("\n");

    printf("AS: ");
    for (int i = 0; i < csr_matrix->NZ; i++)
        printf("%.1f ", csr_matrix->AS[i]);
    printf("\n");
}