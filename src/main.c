#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>

#include "utils_header/mmio.h"
#include "data_structures/ellpack_matrix.h"
#include "data_structures/csr_matix.h"
#include "headers/csr_headers.h"
#include "headers/ellpack_headers.h"
#include "headers/reader_matrix.h"

int main()
{

    int matrix_counter = 0; // Number of matrices processed

    const char *dir_name = "matrici";
    char matrix_filename[256]; // Matrix filename
    char matrix_fullpath[256]; // Buffer to full path of the matrix file to open
    DIR *dir;
    struct dirent *entry;

    csr_matrix matrix;

    // Open dir matrix to take the tests matrix
    dir = opendir(dir_name);
    if (dir == NULL)
    {
        printf("Error occour while opening the matrix directory!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }
    // Take one test matrix every time and transform it to csr and ellapack format
    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_name[0] == '.')
            continue;

        matrix_counter++;
        strcpy(matrix_filename, &entry->d_name);
        printf("Test matrix file: %s\n", matrix_filename);

        // Start to read matrix in the Matrix Market file
        MM_typecode matcode;

        // Create the fullpath for open the matrix file in Matrix Market format
        snprintf(matrix_fullpath, sizeof(matrix_fullpath), "matrici/%s", matrix_filename);

        FILE *matrix_file = fopen(matrix_fullpath, "r");

        if (matrix_file == NULL)
        {
            printf("Error occour while open matrix file!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        if (mm_read_banner(matrix_file, &matcode) != 0)
        {
            printf("Error while try to read banner of Matrix Market for %s!\nError code: %d\n", matrix_filename, errno);
            fclose(matrix_file);
            exit(EXIT_FAILURE);
        }

        if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode))
        {
            printf("The file does not respect the format of sparse matrices!\nError code: %d\n", errno);
            fclose(matrix_file);
            exit(EXIT_FAILURE);
        }

        // Read matrix dimension and convert in csr format
        if (mm_read_mtx_crd_size(matrix_file, &matrix.M, &matrix.N, &matrix.NZ) != 0)
        {
            printf("Error occour while try to read matrix dimension!\nError code: %d\n", errno);
            fclose(matrix_file);
            exit(EXIT_FAILURE);
        }

        puts("");
        printf("Matrix number of columns:           %d\n", matrix.M);
        printf("Matrix number of raws:              %d\n", matrix.N);
        printf("matrix number of non-zero values:   %d\n", matrix.NZ);
        puts("");

        // Allocate arrays to read values from the matrix file
        matrix.raw_indexes = (int *)malloc(matrix.NZ * sizeof(int));
        if (matrix.raw_indexes == NULL)
        {
            printf("Error in malloc for rax indexes array!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        matrix.columns_indexes = (int *)malloc(matrix.NZ * sizeof(int));
        if (matrix.columns_indexes == NULL)
        {
            printf("Error in malloc for columns indexes array!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }
        matrix.readed_values = (double *)malloc(matrix.NZ * sizeof(double));
        if (matrix.readed_values == NULL)
        {
            printf("Error in malloc for non-zero values array!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < matrix.NZ; i++)
        {
            if (fscanf(matrix_file, "%d %d %lf", &matrix.raw_indexes[i], &matrix.columns_indexes[i], &matrix.readed_values[i]) != 3)
            {
                if (fscanf(matrix_file, "%d %d", &matrix.raw_indexes[i], &matrix.columns_indexes[i]) == 2)
                {
                    matrix.readed_values[i] = 1.0;
                }
                else
                {
                    printf("Error occour while trying to read matrix elements!\nError code: %d\n", errno);
                    free(matrix.raw_indexes);
                    free(matrix.columns_indexes);
                    free(matrix.readed_values);
                    exit(EXIT_FAILURE);
                }
            }
            // Back to index matrix to 0
            matrix.raw_indexes[i]--;
            matrix.columns_indexes[i]--;

            // printf("matrix.raw_indexes[%d] = %d.\n", i, matrix.raw_indexes[i]);
            // printf("matrix.columns_indexes[%d] = %d.\n", i, matrix.columns_indexes[i]);
            // printf("mtrix.readed_values[%d] = %.16lf\n", i, matrix.AS[i]);

            sleep(1);
        }
        fclose(matrix_file);

        sleep(1);
    }

    closedir(dir);
    return 0;
}