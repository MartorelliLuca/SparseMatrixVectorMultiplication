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

// Function to read matrix stored and convert it to csr_matrix format
void read_CSR_matrix(FILE *matrix_file, CSR_matrix *csr_matrix, matrix_format *matrix)
{

    if (mm_read_banner(matrix_file, &matrix->matrix_typecode) != 0)
    {
        printf("Error in read of bunner of Matrix Market!\n");
        fclose(matrix_file);
    }

    int is_sparse = mm_is_sparse(matrix->matrix_typecode);
    int is_symmetric = mm_is_symmetric(matrix->matrix_typecode);
    csr_matrix->is_symmetric = is_symmetric;
    int is_pattern = mm_is_pattern(matrix->matrix_typecode);
    int is_array = mm_is_array(matrix->matrix_typecode);

    if (!mm_is_matrix(matrix->matrix_typecode) || (!mm_is_real(matrix->matrix_typecode) && !is_pattern))
    {
        printf("File format non supported\n");
        fclose(matrix_file);
        return NULL;
    }

    if (mm_read_mtx_crd_size(matrix_file, &matrix->M, &matrix->N, &matrix->number_of_non_zeoroes_values) != 0)
    {
        printf("Error in read of dimesion of matrix!\n");
        fclose(matrix_file);
        return NULL;
    }

    int max_entries = is_symmetric ? 2 * matrix->number_of_non_zeoroes_values : matrix->number_of_non_zeoroes_values;
    matrix->rows = malloc(max_entries * sizeof(int));
    matrix->columns = malloc(max_entries * sizeof(int));
    matrix->values = is_pattern ? NULL : malloc(max_entries * sizeof(double));

    if (!matrix->rows || !matrix->columns || (!is_pattern && !matrix->rows))
    {
        printf("Errror in malloc for read CSR matrix!\n");
        free(matrix->rows);
        free(matrix->columns);
        free(matrix->values);
        fclose(matrix_file);
    }

    int count = 0;
    for (int i = 0; i < matrix->number_of_non_zeoroes_values; i++)
    {
        int row, col;
        double val = 1.0;
        if (is_pattern)
        {
            fscanf(matrix_file, "%d %d", &row, &col);
        }
        else
        {
            fscanf(matrix_file, "%d %d %lf", &row, &col, &val);
        }
        row--;
        col--;

        matrix->rows[count] = row;
        matrix->columns[count] = col;
        if (!is_pattern)
            matrix->values[count] = val;
        count++;

        if (is_symmetric && row != col)
        {
            matrix->rows[count] = col;
            matrix->columns[count] = row;
            if (!is_pattern)
                matrix->values[count] = val;
            count++;
        }
    }
    fclose(matrix_file);

    int *IRP = calloc(matrix->M + 1, sizeof(int));
    int *JA = malloc(count * sizeof(int));
    double *AS = malloc(count * sizeof(double));

    if (!IRP || !JA || !AS)
    {
        printf("Error in malloc in read CSR matrix!\n");
        free(matrix->rows);
        free(matrix->columns);
        free(matrix->columns);
        free(IRP);
        free(JA);
        free(AS);
    }

    for (int i = 0; i < count; i++)
    {
        IRP[matrix->rows[i] + 1]++;
    }
    for (int i = 1; i <= matrix->M; i++)
    {
        IRP[i] += IRP[i - 1];
    }

    int *row_fill = calloc(matrix->M, sizeof(int));
    for (int i = 0; i < count; i++)
    {
        int row = matrix->rows[i];
        int pos = IRP[row] + row_fill[row];
        JA[pos] = matrix->columns[i];
        AS[pos] = is_pattern ? 1.0 : matrix->values[i];
        row_fill[row]++;
    }

    free(row_fill);

    csr_matrix->M = matrix->M;
    csr_matrix->N = matrix->N;
    csr_matrix->NZ = matrix->number_of_non_zeoroes_values;
    csr_matrix->AS = AS;
    csr_matrix->JA = JA;
    csr_matrix->IRP = IRP;
    // print_CSR_matrix(csr_matrix);
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

// Function to print matrix values
void print_CSR_matrix(CSR_matrix *csr_matrix)
{
    if (!csr_matrix)
    {
        printf("CSR Matrix is empty!\n");
        return;
    }

    printf("\nPrint Matrix in CSR format\n");
    printf("Matrix:             %s\n", csr_matrix->name);
    printf("Dimension:          %d x %d\n", csr_matrix->M, csr_matrix->N);
    printf("Non-zero Values:    %d\n", csr_matrix->NZ);

    printf("\nAS (values):\n");
    for (int i = 0; i < csr_matrix->NZ; i++)
    {
        printf("%.10f ", csr_matrix->AS[i]);
    }
    printf("\n");

    printf("\nJA (J indices):\n");
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