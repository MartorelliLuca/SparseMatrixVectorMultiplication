#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

#include "../data_structures/hll_matrix.h"
#include "../utils_header/initialization.h"
#include "../data_structures/csr_matrix.h"
#include "../data_structures/ellpack_block.h"
#include "../headers/hll_headers.h"
#include "../headers/csr_headers.h"
#include "../headers/matrix.h"

// Function to read hll_matrix stored in csr format
void read_HLL_matrix(CSR_matrix *csr_matrix, HLL_matrix *hll_matrix) {
    printf("Starting conversion from CSR to HLL.\n");
    
    hll_matrix->hack_size = 32;
    hll_matrix->M = csr_matrix->M;
    hll_matrix->N = csr_matrix->N;
    hll_matrix->num_blocks = (hll_matrix->M + hll_matrix->hack_size - 1) / hll_matrix->hack_size;
    hll_matrix->blocks = (ELLPACKBlock *)calloc(hll_matrix->num_blocks, sizeof(ELLPACKBlock));

    if (hll_matrix->blocks == NULL) {
        printf("Error allocating memory for ELLPACK blocks in HLL matrix!\n");
        exit(EXIT_FAILURE);
    }

    for (int block = 0; block < hll_matrix->num_blocks; block++) {
        int start_row = block * hll_matrix->hack_size;
        int end_row = (start_row + hll_matrix->hack_size < hll_matrix->M) ? start_row + hll_matrix->hack_size : hll_matrix->M;
        
        int max_nz = 0;
        int *row_nz = (int *)calloc(hll_matrix->hack_size, sizeof(int));
        if (row_nz == NULL) {
            printf("Error allocating memory for row_nz array!\n");
            exit(EXIT_FAILURE);
        }

        for (int i = start_row; i < end_row; i++) {
            int nnz_row = csr_matrix->IRP[i + 1] - csr_matrix->IRP[i];
            row_nz[i - start_row] = nnz_row;
            if (nnz_row > max_nz) max_nz = nnz_row;
        }

        hll_matrix->blocks[block].MAXNZ = max_nz;
        hll_matrix->blocks[block].JA = (int **)calloc(hll_matrix->hack_size, sizeof(int *));
        hll_matrix->blocks[block].AS = (double **)calloc(hll_matrix->hack_size, sizeof(double *));

        if (!hll_matrix->blocks[block].JA || !hll_matrix->blocks[block].AS) {
            printf("Error allocating memory for JA/AS pointers in block %d!\n", block);
            exit(EXIT_FAILURE);
        }

        for (int r = 0; r < hll_matrix->hack_size; r++) {
            hll_matrix->blocks[block].JA[r] = (int *)calloc(max_nz, sizeof(int));
            hll_matrix->blocks[block].AS[r] = (double *)calloc(max_nz, sizeof(double));
            
            if (!hll_matrix->blocks[block].JA[r] || !hll_matrix->blocks[block].AS[r]) {
                printf("Error allocating memory for JA[r] or AS[r] in block %d, row %d!\n", block, r);
                exit(EXIT_FAILURE);
            }
        }

        int *nz_count = (int *)calloc(hll_matrix->hack_size, sizeof(int));
        if (nz_count == NULL) {
            printf("Error allocating memory for nz_count array!\n");
            exit(EXIT_FAILURE);
        }

        for (int i = csr_matrix->IRP[start_row]; i < csr_matrix->IRP[end_row]; i++) {
            int row = csr_matrix->JA[i];
            if (row >= start_row && row < end_row) {
                int local_row = row - start_row;
                int nz_idx = nz_count[local_row]++;
                hll_matrix->blocks[block].JA[local_row][nz_idx] = csr_matrix->JA[i];
                hll_matrix->blocks[block].AS[local_row][nz_idx] = csr_matrix->AS[i];
            }
        }

        free(nz_count);
        free(row_nz);
        printf("Completed block %d/%d.\n", block + 1, hll_matrix->num_blocks);
    }
}

// Function to print hll_matrix
void print_HLL_matrix(HLL_matrix *h_matrix) {}

// Function to destroy ellpack_matrix
void destroy_HLL_matrix(HLL_matrix *h_matrix) {}
