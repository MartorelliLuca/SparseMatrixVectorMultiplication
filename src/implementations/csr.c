#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <string.h>

#include "../utils_header/initialization.h"
#include "../headers/csr_headers.h"
#include "../headers/matrix.h"
#include "../data_structures/csr_matrix.h"

// Function to read matrix stored and convert it to csr_matrix format
void read_CSR_matrix(FILE *matrix_file, CSR_matrix *csr_matrix, int *file_type, matrix_format *matrix)
{
    printf("Starting reading CSR.\n");
    csr_matrix->M = matrix->M;
    csr_matrix->N = matrix->N;
    csr_matrix->NZ = matrix->NZ;

    // Create array for csr_matrix struct

    // Create IRP array for csr_matrix format
    csr_matrix->IRP = (int *)calloc(((csr_matrix->M) + 1), sizeof(int));
    if (csr_matrix->IRP == NULL)
    {
        printf("Error occour in malloc for IRP in csr_matrix conversion!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // Create JA array for csr_matrix format
    csr_matrix->JA = (int *)calloc((csr_matrix->NZ), sizeof(int));
    if (csr_matrix->JA == NULL)
    {
        printf("Error occour in malloc for JA in csr_matrix convertion!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // Create AS array for csr_matrix format
    csr_matrix->AS = (double *)calloc((csr_matrix->NZ), sizeof(double));
    if (csr_matrix->AS == NULL)
    {
        printf("Error occour in malloc for AS in csr_matrix convertion!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // If the matrix is sparse
    if (file_type[0])
    {
        // Count the non zero values for every rows
        for (int i = 0; i < csr_matrix->NZ; i++)
            csr_matrix->IRP[matrix->row_indices[i] + 1]++;

        // Inizialize the IRP vector to the correct values
        for (int i = 1; i <= csr_matrix->M; i++)
            csr_matrix->IRP[i] += csr_matrix->IRP[i - 1];

        int *row_positions;
        int row, position;

        // Assign to JA and AS the values
        row_positions = (int *)calloc(csr_matrix->M, sizeof(int));
        if (row_positions == NULL)
        {
            printf("Error occour in malloc for row position in convert to csr_matrix format function!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < csr_matrix->NZ; i++)
        {
            row = matrix->row_indices[i];
            position = csr_matrix->IRP[row] + row_positions[row];
            csr_matrix->JA[position] = matrix->columns_indices[i];
            csr_matrix->AS[position] = matrix->values[i];
            row_positions[row]++;
        }

        free(row_positions);
    }
    else if (file_type[1])
    {
        int index = 0;
        double value = 0.0;
        for (int i = 0; i < csr_matrix->M; i++)
        {
            for (int j = 0; j < csr_matrix->M; j++)
            {
                fscanf(matrix_file, "%lf", &value);
                if (value != 0)
                {
                    csr_matrix->AS[index] = value;
                    csr_matrix->JA[index] = j;
                    index++;
                }
            }
            csr_matrix->IRP[i + 1] = index;
        }
        csr_matrix->NZ = index;
    }

    // print_matrix(csr_matrix);
}

// Function to destroy matrix
void destroy_CSR_matrix(CSR_matrix *csr_matrix) {}

// Function to print matrix values
void print_csr(CSR_matrix *csr_matrix)
{
    if (!csr_matrix)
    {
        printf("CSR Matrix is empty!\n");
        return;
    }

    printf("Dimension: %d x %d\n", csr_matrix->M, csr_matrix->N);
    printf("Non-zero Values: %d\n", csr_matrix->NZ);

    printf("\nAS (values):\n");
    for (int i = 0; i < csr_matrix->NZ; i++)
    {
        printf("%.10f ", csr_matrix->AS[i]);
    }
    printf("\n");

    printf("\nJA (columns indices):\n");
    for (int i = 0; i < csr_matrix->NZ; i++)
    {
        printf("%d ", csr_matrix->JA[i]);
    }
    printf("\n");

    printf("\nIRP (row pointers):\n");
    for (int i = 0; i <= csr_matrix->M; i++)
    {
        printf("%d ", csr_matrix->IRP[i]);
    }
    printf("\n");
}