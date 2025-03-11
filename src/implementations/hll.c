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

void read_HLL_matrix(CSR_matrix *csr_matrix, HLL_matrix *hll_matrix)
{
    if (!csr_matrix || !hll_matrix)
    {
        printf("Error: one of csr or hll pointer is null\n");
        exit(EXIT_FAILURE);
    }

    int M = csr_matrix->M, N = csr_matrix->N;
    int num_hacks = (M + HACKSIZE - 1) / HACKSIZE;

    hll_matrix->hack_offsets = (int *)malloc((num_hacks + 1) * sizeof(int));
    if (!hll_matrix->hack_offsets)
    {
        printf("Error allocating hack_offsets\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    int total_nnz = 0;
    int *maxNR_per_hack = (int *)calloc(num_hacks, sizeof(int));
    if (!maxNR_per_hack)
    {
        printf("Error allocating maxNR_per_hack\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    for (int h = 0; h < num_hacks; h++)
    {
        int start_row = h * HACKSIZE;
        int end_row = (start_row + HACKSIZE < M) ? start_row + HACKSIZE : M;

        int maxNR = 0;
        for (int i = start_row; i < end_row; i++)
        {
            if (csr_matrix->IRP[i] >= csr_matrix->IRP[i + 1])
            {
                printf("Error: IRP not valid in row %d\n", i);
                exit(EXIT_FAILURE);
            }
            int nnz_row = csr_matrix->IRP[i + 1] - csr_matrix->IRP[i];
            if (nnz_row > maxNR)
                maxNR = nnz_row;
        }
        maxNR_per_hack[h] = maxNR;
        total_nnz += (end_row - start_row) * maxNR;
    }

    hll_matrix->AS = (double *)calloc(total_nnz, sizeof(double));
    hll_matrix->JA = (int *)malloc(total_nnz * sizeof(int));
    if (!hll_matrix->AS || !hll_matrix->JA)
    {
        printf("Error allocating AS or JA\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    hll_matrix->M = M;
    hll_matrix->N = N;
    hll_matrix->hack_size = HACKSIZE;
    hll_matrix->num_hacks = num_hacks;

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

            int last_col = -1;
            for (int j = 0; j < row_nnz; j++)
            {
                int idx = csr_matrix->IRP[i] + j;
                hll_matrix->AS[row_offset + j] = csr_matrix->AS[idx];
                hll_matrix->JA[row_offset + j] = last_col = csr_matrix->JA[idx];
            }

            for (int j = row_nnz; j < maxNR; j++)
            {
                hll_matrix->JA[row_offset + j] = (last_col != -1) ? last_col : 0;
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
