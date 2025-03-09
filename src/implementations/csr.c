#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <string.h>
#include "../headers/csr_headers.h"
#include "../data_structures/csr_matix.h"

// Function to read matrix stored and convert it to csr_matrix format
void read_CSR_matrix(FILE *matrix_file, CSR_matrix *csr_matrix, int *file_type)
{
    int result1, result2;
    // Read matrix dimension and convert in csr_matrix format
    if (file_type[0])
    {
        if (mm_read_mtx_crd_size(matrix_file, &csr_matrix->M, &csr_matrix->N, &csr_matrix->NZ) != 0)
        {
            printf("Error occour while try to read matrix dimension!\nError code: %d\n", errno);
            fclose(matrix_file);
            exit(EXIT_FAILURE);
        }
    }
    else if (file_type[1])
    {
        if (mm_read_mtx_array_size(matrix_file, &csr_matrix->M, &csr_matrix->N) != 0)
        {
            printf("Error occour while try to read matrix array dimension!\nError code: %d\n", errno);
            fclose(matrix_file);
            exit(EXIT_FAILURE);
        }

        // If the matrix is an array, the matrix is dense
        csr_matrix->NZ = csr_matrix->M * csr_matrix->N;
    }
    else
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
        // Allocate arrays to read values from the matrix file

        // Allocare array to row indices
        csr_matrix->row_indices = (int *)calloc(csr_matrix->NZ, sizeof(int));
        if (csr_matrix->row_indices == NULL)
        {
            printf("Error in malloc for rax indexes array!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        // Allocate array for columns indices
        csr_matrix->columns_indices = (int *)calloc(csr_matrix->NZ, sizeof(int));
        if (csr_matrix->columns_indices == NULL)
        {
            printf("Error in malloc for columns indexes array!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        // Allocare array for readed values from file
        csr_matrix->readed_values = (double *)calloc(csr_matrix->NZ, sizeof(double));
        if (csr_matrix->readed_values == NULL)
        {
            printf("Error in malloc for non-zero values array!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        // Read values from matrix market file format
        for (int i = 0; i < csr_matrix->NZ; i++)
        {
            // If we can read 3 values then the matrix is not in pattern form
            result1 = fscanf(matrix_file, "%d %d %lf", &csr_matrix->row_indices[i], &csr_matrix->columns_indices[i], &csr_matrix->readed_values[i]);
            if (result1 != 3)
            {
                // If we read 2 values the matrix is in pattern form
                result2 = fscanf(matrix_file, "%d %d", &csr_matrix->row_indices[i], &csr_matrix->columns_indices[i]);
                if (result2 == 2)
                {
                    csr_matrix->readed_values[i] = 1.0;
                }
                else
                {
                    // Debug print
                    // printf("result1 = %d\n", result1);
                    // printf("result2 = %d\n", result2);
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

        // Count the non zero values for every rows
        for (int i = 0; i < csr_matrix->NZ; i++)
            csr_matrix->IRP[csr_matrix->row_indices[i] + 1]++;

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
            row = csr_matrix->row_indices[i];
            position = csr_matrix->IRP[row] + row_positions[row];
            csr_matrix->JA[position] = csr_matrix->columns_indices[i];
            csr_matrix->AS[position] = csr_matrix->readed_values[i];
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