#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

#include "../data_structures/hll_matrix.h"
#include "../data_structures/ellpack_block.h"
#include "../headers/hll_headers.h"
#include "../headers/matrix.h"

// Function to read hll_matrix stored in csr format
void *read_HLL_matrix(FILE *matrix_file, HLL_matrix *hll_matrix, int *file_type, matrix_format *matrix)
{
    hll_matrix->hack_size = 32;
    hll_matrix->M = matrix->M;
    hll_matrix->N = matrix->N;
    hll_matrix->num_blocks = (hll_matrix->M + hll_matrix->hack_size - 1) / hll_matrix->hack_size;
    hll_matrix->blocks = (ELLPACKBlock *)calloc(hll_matrix->num_blocks, sizeof(ELLPACKBlock *));

    if (hll_matrix->blocks == NULL)
    {
        printf("Error occour in malloc for ellpack block in read hll matrix!\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    for (int block = 0; block < hll_matrix->num_blocks; block++)
    {
        int start_row = block * hll_matrix->hack_size;
        int end_row = (block + 1) * hll_matrix->hack_size;
        if (end_row > hll_matrix->M)
            end_row = hll_matrix->M;

        int max_nz = 0;
        int *row_nz = (int *)calloc(hll_matrix->hack_size, sizeof(int));
        if (row_nz == NULL)
        {
            printf("Error occour in malloc for row_nz in read hll matrix!\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < matrix->NZ; i++)
        {
            int r = matrix->row_indices[i];
            if (r >= start_row && r < end_row)
            {
                row_nz[r - start_row]++;
                if (row_nz[r - start_row] > max_nz)
                    max_nz = row_nz[r - start_row];
            }
        }

        hll_matrix->blocks[block].MAXNZ = max_nz;
        hll_matrix->blocks[block].JA = (int **)calloc(hll_matrix->hack_size, sizeof(int *));
        if (hll_matrix->blocks[block].JA == NULL)
        {
            printf("Error occour in malloc for hll_matrix->blocks[block].JA in read hll matrix!\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        hll_matrix->blocks[block].AS = (double **)calloc(hll_matrix->hack_size, sizeof(double *));
        if (hll_matrix->blocks[block].AS == NULL)
        {
            printf("Error occour in malloc for hll_matrix->blocks[block].AS in read hll matrix!\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        for (int r = 0; r < hll_matrix->hack_size; r++)
        {
            hll_matrix->blocks[block].JA[r] = (int *)calloc(max_nz, sizeof(int));
            if (hll_matrix->blocks[block].JA[r] == NULL)
            {
                printf("Error occour in malloc for hll_matrix->blocks[block].JA[r] in read hll matrix!\nError Code: %d\n", errno);
                exit(EXIT_FAILURE);
            }

            hll_matrix->blocks[block].AS[r] = (double *)calloc(max_nz, sizeof(double));
            if (hll_matrix->blocks[block].AS[r] == NULL)
            {
                printf("Error occour in malloc for hll_matrix->blocks[block].AS[r] in read hll matrix!\nError Code: %d\n", errno);
                exit(EXIT_FAILURE);
            }
        }

        int *nz_count = (int *)calloc(hll_matrix->hack_size, sizeof(int));
        if (nz_count == NULL)
        {
            printf("Error occour in malloc for nz_count in read hll matrix!\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < matrix->NZ; i++)
        {
            int r = matrix->row_indices[i] - start_row;
            if (r >= 0 && r < hll_matrix->hack_size)
            {
                int nz_idx = nz_count[r]++;
                hll_matrix->blocks[r].JA[r][nz_idx] = matrix->columns_indices[i];
                hll_matrix->blocks[r].AS[r][nz_idx] = matrix->values[i];
            }
        }
        free(nz_count);
    }
}

// Function to print hll_matrix
void print_HLL_matrix(HLL_matrix *h_matrix) {}

// Function to destroy ellpack_matrix
void destroy_HLL_matrix(HLL_matrix *h_matrix) {}
