#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include "../headers/csr_headers.h"
#include "../data_structures/csr_matix.h"

// Function to read matrix stored in csr format
void read_CSR_matrix(FILE *matrix_file, CSR_matrix *csr_matrix)
{
    // Read matrix dimension and convert in csr format
    if (mm_read_mtx_crd_size(matrix_file, &csr_matrix->M, &csr_matrix->N, &csr_matrix->NZ) != 0)
    {
        printf("Error occour while try to read matrix dimension!\nError code: %d\n", errno);
        fclose(matrix_file);
        exit(EXIT_FAILURE);
    }

    puts("");
    printf("Matrix number of columns:           %d\n", csr_matrix->M);
    printf("Matrix number of raws:              %d\n", csr_matrix->N);
    printf("matrix number of non-zero values:   %d\n", csr_matrix->NZ);
    puts("");

    // Allocate arrays to read values from the matrix file
    csr_matrix->row_indices = (int *)malloc(csr_matrix->NZ * sizeof(int));
    if (csr_matrix->row_indices == NULL)
    {
        printf("Error in malloc for rax indexes array!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    csr_matrix->columns_indices = (int *)malloc(csr_matrix->NZ * sizeof(int));
    if (csr_matrix->columns_indices == NULL)
    {
        printf("Error in malloc for columns indexes array!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }
    csr_matrix->readed_values = (double *)malloc(csr_matrix->NZ * sizeof(double));
    if (csr_matrix->readed_values == NULL)
    {
        printf("Error in malloc for non-zero values array!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < csr_matrix->NZ; i++)
    {
        if (fscanf(matrix_file, "%d %d %lf", &csr_matrix->row_indices[i], &csr_matrix->columns_indices[i], &csr_matrix->readed_values[i]) != 3)
        {
            if (fscanf(matrix_file, "%d %d", &csr_matrix->row_indices[i], &csr_matrix->columns_indices[i]) == 2)
            {
                csr_matrix->readed_values[i] = 1.0;
            }
            else
            {
                printf("Error occour while trying to read matrix elements!\nError code: %d\n", errno);
                free(csr_matrix->row_indices);
                free(csr_matrix->columns_indices);
                free(csr_matrix->readed_values);
                exit(EXIT_FAILURE);
            }
        }
        // Back to index matrix to 0
        csr_matrix->row_indices[i]--;
        csr_matrix->columns_indices[i]--;
    }

    convert_to_csr(csr_matrix);
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

// Function to destroy matrix
void destroy_CSR_matrix(CSR_matrix *csr_matrix) {}

// Function to print matrix values
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