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

FILE *get_matrix_file(const char *dir_name, char *matrix_filename)
{
    // file_type[0] -> is_sparse_matrix
    // file_type[1] -> is_array_file

    // Varibles to read the matrix market file
    MM_typecode matcode;
    char matrix_fullpath[256];
    int is_sparse_matrix, is_array_file;

    // Get the full path of the matrix market file to open
    snprintf(matrix_fullpath, sizeof(matrix_fullpath), "%s/%s", dir_name, matrix_filename);

    // Open the file
    FILE *matrix_file = fopen(matrix_fullpath, "r");
    if (matrix_file == NULL)
    {
        printf("Error occurred while opening matrix file: %s\nError code: %d\n", matrix_filename, errno);
        exit(EXIT_FAILURE);
    }

    return matrix_file;
}

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

void get_matrix_format(FILE *matrix_file, int *file_type, matrix_format *matrix)
{

    int result1, result2;
    // Read matrix dimension and convert in csr_matrix format
    if (file_type[0])
    {
        if (mm_read_mtx_crd_size(matrix_file, &matrix->M, &matrix->N, &matrix->number_of_non_zeoroes_values) != 0)
        {
            printf("Error occour while try to read matrix dimension!\nError code: %d\n", errno);
            fclose(matrix_file);
            exit(EXIT_FAILURE);
        }
    }
    else if (file_type[1])
    {
        if (mm_read_mtx_array_size(matrix_file, &matrix->M, &matrix->N) != 0)
        {
            printf("Error occour while try to read matrix array dimension!\nError code: %d\n", errno);
            fclose(matrix_file);
            exit(EXIT_FAILURE);
        }

        // If the matrix is an array, the matrix is dense
        matrix->number_of_non_zeoroes_values = matrix->M * matrix->N;
    }
    else
    {
        printf("Error occour while try to read matrix dimension!\nError code: %d\n", errno);
        fclose(matrix_file);
        exit(EXIT_FAILURE);
    }

    puts("");
    printf("Matrix number of columns:           %d\n", matrix->M);
    printf("Matrix number of raws:              %d\n", matrix->N);
    printf("matrix number of non-zero values:   %d\n", matrix->number_of_non_zeoroes_values);
    puts("");

    // Allocare array to row indices
    matrix->rows = (int *)calloc(matrix->number_of_non_zeoroes_values, sizeof(int));
    if (matrix->rows == NULL)
    {
        printf("Error in malloc for rax indexes array!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // Allocate array for columns indices
    matrix->columns = (int *)calloc(matrix->number_of_non_zeoroes_values, sizeof(int));
    if (matrix->columns == NULL)
    {
        printf("Error in malloc for columns indexes array!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // Allocare array for readed values from file
    matrix->values = (double *)calloc(matrix->number_of_non_zeoroes_values, sizeof(double));
    if (matrix->values == NULL)
    {
        printf("Error in malloc for non-zero values array!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // Read values from matrix market file format
    for (int i = 0; i < matrix->number_of_non_zeoroes_values; i++)
    {
        // If we can read 3 values then the matrix is not in pattern form
        result1 = fscanf(matrix_file, "%d %d %lf", &matrix->rows[i], &matrix->columns[i], &matrix->values[i]);
        if (result1 != 3)
        {
            // If we read 2 values the matrix is in pattern form
            result2 = fscanf(matrix_file, "%d %d", &matrix->rows[i], &matrix->columns[i]);
            if (result2 == 2)
            {
                matrix->values[i] = 1.0;
            }
            else
            {
                // Debug print
                // printf("result1 = %d\n", result1);
                // printf("result2 = %d\n", result2);
                printf("Error occour while trying to read matrix elements!\nError code: %d\n", errno);
                free(matrix->rows);
                free(matrix->columns);
                free(matrix->values);
                exit(EXIT_FAILURE);
            }
        }
        // Back to index matrix to 0
        matrix->rows[i]--;
        matrix->columns[i]--;
    }

    return &matrix;
}