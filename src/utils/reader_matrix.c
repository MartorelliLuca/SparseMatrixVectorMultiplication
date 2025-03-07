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

// Function to convert matrix in HLL format
void convert_to_ellpack(int M, int N, int nz, int *row_indices, int *col_indices, double *values, int **JA, double **AS, int *MAXNZ)
{
    *MAXNZ = 0;

    // Find max number of nonzero values for every raws
    int *row_counts = (int *)calloc(M, sizeof(int));
    for (int i = 0; i < nz; i++)
    {
        row_counts[row_indices[i]]++;
    }
    for (int i = 0; i < M; i++)
    {
        if (row_counts[i] > *MAXNZ)
        {
            *MAXNZ = row_counts[i];
        }
    }

    // Create JA and AS
    *JA = (int *)malloc(M * (*MAXNZ) * sizeof(int));
    *AS = (double *)malloc(M * (*MAXNZ) * sizeof(double));

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < *MAXNZ; j++)
        {

            // Check if the index is valid
            (*JA)[i * (*MAXNZ) + j] = (j == 0) ? 1 : (*JA)[i * (*MAXNZ) + j - 1];
            // Zero value for empty cells
            (*AS)[i * (*MAXNZ) + j] = 0.0;
        }
    }

    // Insert values in JA and AS
    int *row_position = (int *)calloc(M, sizeof(int));
    for (int i = 0; i < nz; i++)
    {
        int row = row_indices[i];
        int pos = row_position[row];
        // Index start from 1
        (*JA)[row * (*MAXNZ) + pos] = col_indices[i] + 1;
        (*AS)[row * (*MAXNZ) + pos] = values[i];
        row_position[row]++;
    }

    free(row_counts);
    free(row_position);
}

// Function to convert matrix in CSR format
void convert_to_csr(csr_matrix *matrix)
{
    // Create array for matrix struct
    matrix->IRP = (int *)malloc((matrix->M + 1) * sizeof(int));
    if (matrix->IRP == NULL)
    {
        printf("Error occour in malloc for IRP in csr conversion!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    matrix->JA = (int *)malloc((matrix->NZ) * sizeof(int));
    if (matrix->JA == NULL)
    {
        printf("Error occour in malloc for JA in csr convertion!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    matrix->AS = (double *)malloc((matrix->NZ) * sizeof(double));
    if (matrix->AS == NULL)
    {
        printf("Error occour in malloc for AS in csr convertion!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // Initialize IRP array to 0
    for (int i = 0; i <= matrix->M; i++)
        matrix->IRP[i] = 0;
}
// void convert_to_csr(int M, int N, int nz, int *row_indices, int *col_indices, double *values, int **IRP, int **JA, double **AS)
// {
//     *IRP = (int *)malloc((M + 1) * sizeof(int)); // Arrays of pointer to raws
//     *JA = (int *)malloc(nz * sizeof(int));       // Arrays of columns indexes
//     *AS = (double *)malloc(nz * sizeof(double)); // Arrays of non null values

//     // Inizialize IRP to zero
//     for (int i = 0; i <= M; i++)
//     {
//         (*IRP)[i] = 0;
//     }

//     // Count the non zero values for every raws
//     for (int i = 0; i < nz; i++)
//     {
//         (*IRP)[row_indices[i] + 1]++;
//     }

//     // Create IRP
//     for (int i = 0; i < M; i++)
//     {
//         (*IRP)[i + 1] += (*IRP)[i];
//     }

//     // Assign to JA and to AS the values
//     int *row_position = (int *)calloc(M, sizeof(int));
//     for (int i = 0; i < nz; i++)
//     {
//         int row = row_indices[i];
//         int pos = (*IRP)[row] + row_position[row];
//         (*JA)[pos] = col_indices[i];
//         (*AS)[pos] = values[i];
//         row_position[row]++;
//     }

//     free(row_position);
// }

// First attempt to do matrix-vector dot product in CSR format
void matvec_csr(int M, int *IRP, int *JA, double *AS, double *x, double *y)
{
    for (int i = 0; i < M; i++)
    {
        y[i] = 0.0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++)
        {
            y[i] += AS[j] * x[JA[j]];
        }
    }
}

// First attempt to do matrix-vector dot produt in ELLPACK format
void matvec_ellpack(int M, int MAXNZ, int *JA, double *AS, double *x, double *y)
{
    for (int i = 0; i < M; i++)
    {
        y[i] = 0.0;
        for (int j = 0; j < MAXNZ; j++)
        {
            int col = JA[i * MAXNZ + j];
            // We have to check if the column index is valid
            if (col != -1)
            {
                y[i] += AS[i * MAXNZ + j] * x[col];
            }
        }
    }
}