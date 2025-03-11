#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <string.h>

#include "../utils_header/initialization.h"
#include "../headers/csr_headers.h"
#include "../headers/matrix.h"
#include "../data_structures/csr_matrix.h"
#include "../utils_header/mmio.h"

// Function to read matrix stored and convert it to csr_matrix format
void read_CSR_matrix(FILE *matrix_file, CSR_matrix *csr_matrix)
{

    MM_typecode matcode;
    if (mm_read_banner(matrix_file, &matcode) != 0)
    {
        printf("Error in read of bunner of Matrix Market!\n");
        fclose(matrix_file);
        return NULL;
    }

    int is_sparse = mm_is_sparse(matcode);
    int is_symmetric = mm_is_symmetric(matcode);
    csr_matrix->is_symmetric = is_symmetric;
    int is_pattern = mm_is_pattern(matcode);
    int is_array = mm_is_array(matcode);

    if (!mm_is_matrix(matcode) || (!mm_is_real(matcode) && !is_pattern))
    {
        printf("File format non supported\n");
        fclose(matrix_file);
        return NULL;
    }

    int M, N, NZ;
    if (mm_read_mtx_crd_size(matrix_file, &M, &N, &NZ) != 0)
    {
        printf("Error in read of dimesion of matrix!\n");
        fclose(matrix_file);
        return NULL;
    }

    int max_entries = is_symmetric ? 2 * NZ : NZ;
    int *I = malloc(max_entries * sizeof(int));
    int *J = malloc(max_entries * sizeof(int));
    double *values = is_pattern ? NULL : malloc(max_entries * sizeof(double));

    if (!I || !J || (!is_pattern && !values))
    {
        printf("Errror in malloc for read CSR matrix!\n");
        free(I);
        free(J);
        free(values);
        fclose(matrix_file);
        return NULL;
    }

    int count = 0;
    for (int i = 0; i < NZ; i++)
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

        I[count] = row;
        J[count] = col;
        if (!is_pattern)
            values[count] = val;
        count++;

        if (is_symmetric && row != col)
        {
            I[count] = col;
            J[count] = row;
            if (!is_pattern)
                values[count] = val;
            count++;
        }
    }
    fclose(matrix_file);

    int *IRP = calloc(M + 1, sizeof(int));
    int *JA = malloc(count * sizeof(int));
    double *AS = malloc(count * sizeof(double));

    if (!IRP || !JA || !AS)
    {
        printf("Error in malloc in read CSR matrix!\n");
        free(I);
        free(J);
        free(values);
        free(IRP);
        free(JA);
        free(AS);
        return NULL;
    }

    for (int i = 0; i < count; i++)
    {
        IRP[I[i] + 1]++;
    }
    for (int i = 1; i <= M; i++)
    {
        IRP[i] += IRP[i - 1];
    }

    int *row_fill = calloc(M, sizeof(int));
    for (int i = 0; i < count; i++)
    {
        int row = I[i];
        int pos = IRP[row] + row_fill[row];
        JA[pos] = J[i];
        AS[pos] = is_pattern ? 1.0 : values[i];
        row_fill[row]++;
    }

    free(I);
    free(J);
    free(values);
    free(row_fill);

    csr_matrix->M = M;
    csr_matrix->N = N;
    csr_matrix->NZ = count;
    csr_matrix->AS = AS;
    csr_matrix->JA = JA;
    csr_matrix->IRP = IRP;
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

    printf("Dimension: %d x %d\n", csr_matrix->M, csr_matrix->N);
    printf("Non-zero Values: %d\n", csr_matrix->NZ);

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