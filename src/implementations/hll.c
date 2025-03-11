#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

#include "../data_structures/hll_matrix.h"
#include "../utils_header/initialization.h"
#include "../data_structures/csr_matrix.h"
#include "../headers/hll_headers.h"
#include "../headers/csr_headers.h"
#include "../headers/matrix.h"

// Function to read hll_matrix stored in csr format
void read_HLL_matrix(CSR_matrix *csr_matrix, HLL_matrix *hll_matrix)
{
    int M = csr_matrix->M, N = csr_matrix->N;
    int num_hacks = (M + HACKSIZE - 1) / HACKSIZE;

    hll_matrix->hack_offsets = (int *)malloc((num_hacks + 1) * sizeof(int));
    if (hll_matrix->hack_offsets == NULL)
    {
        printf("Error in malloc in read hll matrix!\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    int total_nnz = 0;
    int *maxNR_per_hack = (int *)malloc(num_hacks * sizeof(int));

    for (int h = 0; h < num_hacks; h++)
    {
        int start_row = h * HACKSIZE;
        int end_row = (start_row + HACKSIZE < M) ? start_row + HACKSIZE : M;

        int maxNR = 0;
        for (int i = start_row; i < end_row; i++)
        {
            int nnz_row = csr_matrix->IRP[i + 1] - csr_matrix->IRP[i];
            if (nnz_row > maxNR)
                maxNR = nnz_row;
        }
        maxNR_per_hack[h] = maxNR;
        total_nnz += (end_row - start_row) * maxNR;
    }

    hll_matrix->AS = (double *)calloc(total_nnz, sizeof(double));
    if (hll_matrix->AS == NULL)
    {
        printf("Error in malloc in read hll matrix!\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    hll_matrix->JA = (int *)malloc(total_nnz * sizeof(int));
    if (hll_matrix->JA == NULL)
    {
        printf("Error in malloc in read hll matrix!\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < total_nnz; i++)
    {
        hll_matrix->JA[i] = -1;
    }

    int offset = 0;
    for (int h = 0; h < num_hacks; h++)
    {
        hll_matrix->hack_offsets[h] = offset;
        int start_row = h * HACKSIZE;
        int end_row = (start_row + HACKSIZE < M) ? start_row + HACKSIZE : M;
        int maxNR = maxNR_per_hack[h];

        for (int i = start_row; i < end_row; i++)
        {
            int row_offset = offset + (i - start_row) * maxNR;
            int row_nnz = csr_matrix->IRP[i + 1] - csr_matrix->IRP[i];

            for (int j = 0; j < row_nnz; j++)
            {
                hll_matrix->AS[row_offset + j] = csr_matrix->AS[csr_matrix->IRP[i] + j];
                hll_matrix->JA[row_offset + j] = csr_matrix->JA[csr_matrix->IRP[i] + j];
            }

            for (int j = row_nnz; j < maxNR; j++)
            {
                hll_matrix->JA[row_offset + j] = -1;
            }
        }
        offset += (end_row - start_row) * maxNR;
    }

    hll_matrix->hack_offsets[num_hacks] = offset;

    free(maxNR_per_hack);
}

// Function to print hll_matrix
void print_HLL_matrix(HLL_matrix *hll_matrix)
{
    printf("HLL Matrix Representation:\n");
    printf("Rows: %d, Columns: %d, HackSize: %d, NumHacks: %d\n",
           hll_matrix->M, hll_matrix->N, hll_matrix->hack_size, hll_matrix->num_hacks);

    printf("Hack Offsets: ");
    for (int i = 0; i <= hll_matrix->num_hacks; i++)
    {
        printf("%d ", hll_matrix->hack_offsets[i]);
    }
    printf("\n");

    printf("JA (Column Indices):\n");
    for (int i = 0; i < hll_matrix->hack_offsets[hll_matrix->num_hacks]; i++)
    {
        printf("%d ", hll_matrix->JA[i]);
    }
    printf("\n");

    printf("AS (Values):\n");

    for (int i = 0; i < hll_matrix->hack_offsets[hll_matrix->num_hacks]; i++)
    {
        printf("%.2f ", hll_matrix->AS[i]);
    }
    printf("\n");
}

// Function to destroy ellpack_matrix
void destroy_HLL_matrix(HLL_matrix *hll_matrix)
{
    // for (int block = 0; block < hll_matrix->num_blocks; block++)
    // {
    //     for (int r = 0; r < hll_matrix->hack_size; r++)
    //     {
    //         free(hll_matrix->blocks[block].JA[r]);
    //         free(hll_matrix->blocks[block].AS[r]);
    //     }
    // }
    // free(hll_matrix->blocks);
    // free(hll_matrix);

    free(hll_matrix->hack_offsets);
    free(hll_matrix->JA);
    free(hll_matrix->AS);
    free(hll_matrix);
}
