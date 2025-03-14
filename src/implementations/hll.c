#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

#include "../data_structures/hll_matrix.h"
#include "../utils_header/initialization.h"
#include "../data_structures/csr_matrix.h"
#include "../headers/hll_headers.h"
#include "../headers/csr_headers.h"
#include "../headers/matrix_format.h"

// Function to create a matrix in HLL format
void read_HLL_matrix(matrix_format *matrix, HLL_matrix *hll_matrix)
{
    strcpy(hll_matrix->name, matrix->name);
    hll_matrix->M = matrix->M;
    hll_matrix->N = matrix->N;
    hll_matrix->hack_size = HACKSIZE;
    hll_matrix->num_hack = (matrix->M + HACKSIZE - 1) / HACKSIZE;

    int *row_nnz = (int *)calloc(matrix->M, sizeof(int));
    for (int i = 0; i < matrix->number_of_non_zeoroes_values; i++)
    {
        row_nnz[matrix->rows[i]]++;
    }

    int max_non_zeroes_per_block = 0;
    int *max_non_zeroes = (int *)calloc(hll_matrix->num_hack, sizeof(int));
    for (int i = 0; i < hll_matrix->num_hack; i++)
    {
        int local_max = 0;
        for (int j = i * HACKSIZE; j < (i + 1) * HACKSIZE && j < matrix->M; j++)
        {
            if (row_nnz[j] > local_max)
            {
                local_max = row_nnz[j];
            }
        }
        max_non_zeroes[i] = local_max;
        if (local_max > max_non_zeroes_per_block)
        {
            max_non_zeroes_per_block = local_max;
        }
    }

    hll_matrix->max_non_zeroes = max_non_zeroes;
    hll_matrix->num_values = hll_matrix->num_hack * HACKSIZE * max_non_zeroes_per_block;
    hll_matrix->values = (double *)calloc(hll_matrix->num_values, sizeof(double));
    hll_matrix->columns = (int *)calloc(hll_matrix->num_values, sizeof(int));
    hll_matrix->offest = (int *)calloc(hll_matrix->num_hack + 1, sizeof(int));

    for (int i = 0; i < hll_matrix->num_values; i++)
    {
        hll_matrix->columns[i] = -1;
    }

    int *current_pos = (int *)calloc(matrix->M, sizeof(int));
    for (int i = 0; i < matrix->number_of_non_zeoroes_values; i++)
    {
        int row = matrix->rows[i];
        int col = matrix->columns[i];
        double val = matrix->values[i];

        int block_idx = row / HACKSIZE;
        int local_row = row % HACKSIZE;
        int max_cols = max_non_zeroes[block_idx];

        int index = block_idx * HACKSIZE * max_cols + local_row * max_cols + current_pos[row];
        hll_matrix->columns[index] = col;
        hll_matrix->values[index] = val;
        current_pos[row]++;
    }

    int offset_value = 0;
    for (int i = 0; i <= hll_matrix->num_hack; i++)
    {
        hll_matrix->offest[i] = offset_value;
        if (i < hll_matrix->num_hack)
        {
            offset_value += HACKSIZE * max_non_zeroes[i];
        }
    }
    hll_matrix->num_offset = hll_matrix->num_hack + 1;

    free(row_nnz);
    free(current_pos);
}

// Function to print hll_matrix
void print_HLL_matrix(const HLL_matrix *matrix)
{

    printf("Matrix Name: %s\n", matrix->name);
    printf("Dimensions: %d x %d\n", matrix->M, matrix->N);
    printf("HACKSIZE: %d, Num Hacks: %d\n", matrix->hack_size, matrix->num_hack);

    printf("\nOffsets:\n");
    for (int i = 0; i < matrix->num_offset; i++)
    {
        printf("%d ", matrix->offest[i]);
    }
    printf("\n");

    printf("\nValues and Columns:\n");
    int index = 0;
    for (int b = 0; b < matrix->num_hack; b++)
    {
        int max_nz = matrix->max_non_zeroes[b];
        printf("Block %d (Max NZ per row: %d):\n", b, max_nz);
        for (int r = 0; r < matrix->hack_size && (b * matrix->hack_size + r) < matrix->M; r++)
        {
            printf("Row %d: ", b * matrix->hack_size + r);
            for (int c = 0; c < max_nz; c++)
            {
                int col_idx = matrix->columns[index];
                double val = matrix->values[index];
                if (col_idx != -1)
                {
                    printf("(%d, %.2f) ", col_idx, val);
                }
                index++;
            }
            printf("\n");
        }
    }
}

// Function to destroy ellpack_matrix
void destroy_HLL_matrix(HLL_matrix *matrix)
{
    free(matrix->offest);
    free(matrix->columns);
    free(matrix->values);
    free(matrix->max_non_zeroes);
}
