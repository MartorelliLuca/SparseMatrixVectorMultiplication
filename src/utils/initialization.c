#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <time.h>

#include "../utils_header/initialization.h"
#include "../utils_header/mmio.h"
#include "../headers/matrix_format.h"
#include "../data_structures/csr_matrix.h"

double *initialize_x_vector(int size)
{

    // Inizialize the psudo-random number generator with time-based seed
    srand(time(NULL));

    // Create x vector
    double *x = (double *)calloc(size, sizeof(double));
    if (x == NULL)
    {
        printf("Error occur in malloc for the x vector!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // Inizialize x vector to presudo-random double values
    for (int i = 0; i < size; i++)
        x[i] = (rand() % 5) + 1;

    return x;
}

double *initialize_y_vector(int size)
{
    double *y = (double *)calloc(size, sizeof(double));
    if (y == NULL)
    {
        printf("Error occour in malloc for the y vector!\n Error code: %d\n", errno);
        exit(EXIT_FAILURE);
    }
}

void re_initialize_y_vector(int size, double *y)
{
    for (int i = 0; i < size; i++)
        y[i] = 0.0;
}

double **get_matrix_from_csr(CSR_matrix *csr_matrix)
{
    double **dense_matrix = (double **)malloc(csr_matrix->M * sizeof(double *));
    if (!dense_matrix)
    {
        fprintf(stderr, "Errore di allocazione della matrice densa\n");
        return NULL;
    }
    for (int i = 0; i < csr_matrix->M; i++)
    {
        dense_matrix[i] = (double *)calloc(csr_matrix->N, sizeof(double));
        if (!dense_matrix[i])
        {
            fprintf(stderr, "Errore di allocazione della riga %d della matrice densa\n", i);
            for (int j = 0; j < i; j++)
                free(dense_matrix[j]);
            free(dense_matrix);
            return NULL;
        }
    }

    for (int i = 0; i < csr_matrix->M; i++)
    {
        for (int j = csr_matrix->IRP[i]; j < csr_matrix->IRP[i + 1]; j++)
        {
            int col = csr_matrix->JA[j];
            dense_matrix[i][col] = csr_matrix->AS[j];
        }
    }

    return dense_matrix;
}

void free_dense_matrix(double **dense_matrix, int M)
{
    for (int i = 0; i < M; i++)
        free(dense_matrix[i]);
    free(dense_matrix);
}