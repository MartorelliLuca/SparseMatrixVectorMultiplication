#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <time.h>

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

    CSR_matrix csr_matrix;
    ELLPACK_matrix ellpack_matrix;

    // Inizialize the psudo-random number generator with time-based seed
    srand(time(NULL));
    double *x;
    double *y;

    // Variables to collect statistics
    struct timespec start, end;
    double time_used;
    double flops, mflops, gflops;

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
        if (mm_read_mtx_crd_size(matrix_file, &csr_matrix.M, &csr_matrix.N, &csr_matrix.NZ) != 0)
        {
            printf("Error occour while try to read matrix dimension!\nError code: %d\n", errno);
            fclose(matrix_file);
            exit(EXIT_FAILURE);
        }

        puts("");
        printf("Matrix number of columns:           %d\n", csr_matrix.M);
        printf("Matrix number of raws:              %d\n", csr_matrix.N);
        printf("matrix number of non-zero values:   %d\n", csr_matrix.NZ);
        puts("");

        // Allocate arrays to read values from the matrix file
        csr_matrix.row_indices = (int *)malloc(csr_matrix.NZ * sizeof(int));
        if (csr_matrix.row_indices == NULL)
        {
            printf("Error in malloc for rax indexes array!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        csr_matrix.columns_indices = (int *)malloc(csr_matrix.NZ * sizeof(int));
        if (csr_matrix.columns_indices == NULL)
        {
            printf("Error in malloc for columns indexes array!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }
        csr_matrix.readed_values = (double *)malloc(csr_matrix.NZ * sizeof(double));
        if (csr_matrix.readed_values == NULL)
        {
            printf("Error in malloc for non-zero values array!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < csr_matrix.NZ; i++)
        {
            if (fscanf(matrix_file, "%d %d %lf", &csr_matrix.row_indices[i], &csr_matrix.columns_indices[i], &csr_matrix.readed_values[i]) != 3)
            {
                if (fscanf(matrix_file, "%d %d", &csr_matrix.row_indices[i], &csr_matrix.columns_indices[i]) == 2)
                {
                    csr_matrix.readed_values[i] = 1.0;
                }
                else
                {
                    printf("Error occour while trying to read matrix elements!\nError code: %d\n", errno);
                    free(csr_matrix.row_indices);
                    free(csr_matrix.columns_indices);
                    free(csr_matrix.readed_values);
                    exit(EXIT_FAILURE);
                }
            }
            // Back to index matrix to 0
            csr_matrix.row_indices[i]--;
            csr_matrix.columns_indices[i]--;

            // printf("matrix.row_indices[%d] = %d\n", i, csr_matrix.row_indices[i]);
            // printf("matrix.columns_indices[%d] = %d\n", i, csr_matrix.columns_indices[i]);
            // printf("mtrix.readed_values[%d] = %.16lf\n", i, csr_matrix.readed_values[i]);
        }

        convert_to_csr(&csr_matrix);

        // Create x vector
        x = (double *)malloc(csr_matrix.N * sizeof(double));
        if (x == NULL)
        {
            printf("Error occur in malloc for the x vector!\nError code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        y = (double *)malloc(csr_matrix.M * sizeof(double));
        if (y == NULL)
        {
            printf("Error occour in malloc for the y vector!\n Error code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        // Inizialize x vector to presudo-random double values
        for (int i = 0; i < csr_matrix.M; i++)
            x[i] = (rand() % 5) + 1;

        // for (int i = 0; i < csr_matrix.M; i++)
        //     printf("x[%d] = %lf\n", i, x[i]);

        // Inizialize y vector
        for (int i = 0; i < csr_matrix.M; i++)
            y[i] = 0.0;

        printf("\nBefore dot product!\n");
        // Get statistics for the dot-product
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Start dot-product
        matvec_csr(&csr_matrix, x, y);

        // Get the time used for the dot-product
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("After dot product!\n\n\n");

        time_used = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        flops = 2.0 * csr_matrix.NZ / time_used;
        mflops = flops / 1e6;
        gflops = flops / 1e9;

        printf("Time used for dot-product:      %.16lf\n", time_used);
        printf("Performance\n");
        printf("FLOPS:                          %.16lf\n", flops);
        printf("MFLOPS:                         %.16lf\n", mflops);
        printf("GFLOPS:                         %.16lf\n", gflops);

        fclose(matrix_file);
    }

    closedir(dir);
    return 0;
}