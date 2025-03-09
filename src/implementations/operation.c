#include <ctype.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <bits/time.h>
#include <errno.h>

#include "../data_structures/csr_matix.h"
#include "../data_structures/hll_matrix.h"

void matvec(double **A, double *x, double *y, int M, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            y[i] += A[i][j] * x[j];
}

// First attempt to do matrix-vector dot product in CSR format
void matvec_csr(CSR_matrix *csr_matrix, double *x, double *y)
{
#pragma omp parallel for
    for (int i = 0; i < csr_matrix->M; i++)
    {
        for (int j = csr_matrix->IRP[i]; j < csr_matrix->IRP[i + 1]; j++)
        {
            y[i] += csr_matrix->AS[j] * x[csr_matrix->JA[j]];
        }
    }
}

// First attempt to do matrix-vector dot produt in ELLPACK format
void matvec_ellpack(HLL_matrix *ellpack_matrix, double *x, double *y)
{
}

int csr_matrix_vector_product_blocked(CSR_matrix *matrix, double *x, double *y, int block_size)
{
    int new_non_zero_values_count = 0;
    for (int block_start = 0; block_start < matrix->M; block_start += block_size)
    {
        int block_end = (block_start + block_size > matrix->M) ? matrix->M : block_start + block_size;

        // Allocazione temporanea della porzione della matrice
        double **dense_block = (double **)calloc((block_end - block_start), sizeof(double *));
        if (!dense_block)
        {
            printf("Erro in malloc in dot product in memory\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < block_end - block_start; i++)
        {
            dense_block[i] = (double *)calloc(matrix->N, sizeof(double));
            if (!dense_block[i])
            {
                printf("Errore di allocazione della riga %d del blocco\n", errno);
                for (int j = 0; j < i; j++)
                    free(dense_block[j]);
                free(dense_block);
                exit(EXIT_FAILURE);
            }
        }

        // Popoliamo il blocco con i dati CSR
        for (int i = block_start; i < block_end; i++)
        {
            for (int j = matrix->IRP[i]; j < matrix->IRP[i + 1]; j++)
            {
                int col = matrix->JA[j];
                dense_block[i - block_start][col] = matrix->AS[j];
            }
        }

        // Calcoliamo il prodotto della porzione della matrice con il vettore
        for (int i = 0; i < block_end - block_start; i++)
        {
            for (int j = 0; j < matrix->N; j++)
            {
                if (dense_block[i][j] != 0.0)
                {
                    new_non_zero_values_count++;
                    y[block_start + i] += dense_block[i][j] * x[j];
                }
            }
        }

        // Deallocazione del blocco
        for (int i = 0; i < block_end - block_start; i++)
            free(dense_block[i]);
        free(dense_block);

        return new_non_zero_values_count;
    }
}