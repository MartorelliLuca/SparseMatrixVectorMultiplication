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
    int *row_start = (int *)calloc(matrix->M + 1, sizeof(int));
    if (row_start == NULL)
    {
        printf("Error occour in malloc for read hll matrix\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < matrix->number_of_non_zeoroes_values; i++)
        row_start[matrix->rows[i] + 1]++;

    for (int i = 1; i <= matrix->M; i++)
        row_start[i] += row_start[i - 1];

    int *sorted_col_indices = malloc(matrix->number_of_non_zeoroes_values * sizeof(int));
    double *sorted_values = malloc(matrix->number_of_non_zeoroes_values * sizeof(double));
    if (!sorted_col_indices || !sorted_values)
    {
        printf("Error occour in malloc in read hll matrix\nError Code: %d\n", errno);
        free(row_start);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < matrix->number_of_non_zeoroes_values; i++)
    {
        int row = matrix->rows[i];
        int pos = row_start[row]++;
        sorted_col_indices[pos] = matrix->columns[i];
        sorted_values[pos] = matrix->values[i];
    }

    for (int i = matrix->M; i > 0; i--)
        row_start[i] = row_start[i - 1];

    int *nz_per_row = calloc(matrix->M, sizeof(int));
    if (!nz_per_row)
    {
        printf("Error in malloc for read hll matrix\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < matrix->number_of_non_zeoroes_values; i++)
    {
        int row_idx = matrix->rows[i];
        nz_per_row[row_idx]++;
    }

    for (int block_idx = 0; block_idx < hll_matrix->number_of_blocks; block_idx++)
    {
        int start_row = block_idx * HACKSIZE;
        int end_row = (block_idx + 1) * HACKSIZE;
        if (end_row > matrix->M)
            end_row = matrix->M;

        int non_zeroes_per_block = 0;
        for (int i = start_row; i < end_row; i++)
        {
            non_zeroes_per_block += nz_per_row[i];
        }

        int max_non_zeroes_per_row = 0;
        for (int i = start_row; i < end_row; i++)
        {
            if (nz_per_row[i] > max_non_zeroes_per_row)
                max_non_zeroes_per_row = nz_per_row[i];
        }

        hll_matrix->blocks[block_idx].non_zeroes_per_block = non_zeroes_per_block;
        hll_matrix->blocks[block_idx].max_non_zeroes_per_row = max_non_zeroes_per_row;

        int max_nz_per_row = hll_matrix->blocks[block_idx].max_non_zeroes_per_row;
        int rows_in_block = end_row - start_row;
        int size_of_arrays = max_nz_per_row * rows_in_block;
        if (max_nz_per_row < 0 || rows_in_block < 0)
        {
            fprintf(stderr, "Error: invalid value for block %d: %d - %d\n", block_idx, rows_in_block, max_nz_per_row);
            exit(EXIT_FAILURE);
        }

        hll_matrix->blocks[block_idx].JA = calloc(size_of_arrays, sizeof(int));
        hll_matrix->blocks[block_idx].AS = calloc(size_of_arrays, sizeof(double));
        if (!hll_matrix->blocks[block_idx].JA || !hll_matrix->blocks[block_idx].AS)
        {
            fprintf(stderr, "Error occour in malloc for block %d.\n", block_idx);
            for (int k = 0; k < block_idx; k++)
            {
                free(hll_matrix->blocks[k].JA);
                free(hll_matrix->blocks[k].AS);
            }
            free(nz_per_row);
            exit(EXIT_FAILURE);
        }

        memset(hll_matrix->blocks[block_idx].JA, -1, size_of_arrays * sizeof(int));
        memset(hll_matrix->blocks[block_idx].AS, 0, size_of_arrays * sizeof(double));

        for (int i = start_row; i < end_row; i++)
        {
            int row_offset = (i - start_row) * max_nz_per_row;
            int row_nz_start = row_start[i];
            int row_nz_end = row_start[i + 1];

            int pos = 0;
            int last_col_idx = -1;

            for (int j = row_nz_start; j < row_nz_end; j++)
            {
                if (pos >= max_nz_per_row)
                    break;
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = sorted_col_indices[j];
                hll_matrix->blocks[block_idx].AS[index] = sorted_values[j];
                last_col_idx = sorted_col_indices[j];
                pos++;
            }

            while (pos < max_nz_per_row)
            {
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = last_col_idx;
                hll_matrix->blocks[block_idx].AS[index] = 0.0;
                pos++;
            }
        }
    }

    free(row_start);
    free(sorted_col_indices);
    free(sorted_values);
    free(nz_per_row);
}

// Function to print hll_matrix
void print_HLL_matrix(HLL_matrix *hll_matrix)
{
}

// Function to destroy ellpack_matrix
void destroy_HLL_matrix(HLL_matrix *hll_matrix)
{
}
