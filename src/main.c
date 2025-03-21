#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "utils_header/mmio.h"
#include "data_structures/hll_matrix.h"
#include "data_structures/csr_matrix.h"
#include "data_structures/performance.h"
#include "headers/matrix_format.h"
#include "headers/csr_headers.h"
#include "headers/hll_headers.h"
#include "headers/operation.h"
#include "utils_header/initialization.h"
#include "utils_header/utils.h"
#include "utils_header/computation_type.h"
#include "../CUDA_include/cudahll.h"
#include "../CUDA_include/cudacsr.h"

#define MATRIX_DIR = "../matrici"

double *unit_vector(int size)
{
    double *x = (double *)malloc(size * sizeof(double));
    if (!x)
    {
        printf("Error in malloc per x\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++)
        x[i] = 1.0;

    return x;
}

int main()
{
    int *thread_numbers = (int *)calloc(39, sizeof(int));
    if (!thread_numbers)
    {
        printf("Error occour while creating thread_numbers pointer!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    int count = 2;

    for (int i = 0; i < 39; i++)
    {
        thread_numbers[i] = count + i;
    }

    int matrix_counter = 0; // Number of matrices processed

    // Variables to collect statistics
    struct performance *head = NULL, *tail = NULL, *node = NULL;
    double time_used, start, end;
    double flops, mflops, gflops;
    float time;

    const char *dir_name = "../matrici";
    // Matrix filename
    char matrix_filename[256];
    // Buffer to full path of the matrix file to open
    char matrix_fullpath[256];
    DIR *dir;
    struct dirent *entry;
    FILE *matrix_file;
    matrix_format *matrix = NULL;

    CSR_matrix *csr_matrix = NULL;
    HLL_matrix *hll_matrix = NULL;

    double *x, *y, *z;

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

        csr_matrix = (CSR_matrix *)calloc(1, sizeof(CSR_matrix));
        if (csr_matrix == NULL)
        {
            printf("Error occour in malloc for csr matrix\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        hll_matrix = (HLL_matrix *)calloc(1, sizeof(HLL_matrix));
        if (hll_matrix == NULL)
        {
            printf("Error occour in malloc for hll matrix\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        matrix = (matrix_format *)calloc(1, sizeof(matrix_format));
        if (matrix == NULL)
        {
            printf("Error occour in malloc for matrix format\nErro Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        if (entry->d_name[0] == '.')
            continue;

        matrix_counter++;

        strcpy(matrix_filename, entry->d_name);
        strcpy(matrix->name, matrix_filename);

        printf("Processing %s matrix\n", matrix->name);
        matrix_file = get_matrix_file(dir_name, matrix_filename);

        if (read_matrix(matrix_file, matrix) == 1)
        {
            printf("Error in read matrix!\n");
            exit(EXIT_FAILURE);
        }

        printf("Non zeroes values: %d\n", matrix->number_of_non_zeoroes_values);
        // Get matrix from matrix market format in csr format
        strcpy(csr_matrix->name, matrix_filename);
        read_CSR_matrix(matrix_file, csr_matrix, matrix);

        // Get matrix from matrix market format in hll format
        strcpy(hll_matrix->name, matrix_filename);
        read_HLL_matrix(hll_matrix, HACKSIZE, matrix);

        // Initialize vectors
        x = initialize_x_vector(csr_matrix->M);
        y = initialize_y_vector(csr_matrix->M);
        z = initialize_y_vector(csr_matrix->M);

        node = (struct performance *)calloc(1, sizeof(struct performance));
        if (node == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
        }

        strcpy(node->matrix, csr_matrix->name);
        node->non_zeroes_values = matrix->number_of_non_zeoroes_values;
        node->computation = SERIAL_CSR;

        // Get statistics for the dot-product
        start = omp_get_wtime();
        // Start dot-product
        matvec_serial_csr(csr_matrix, x, y);
        // Get the time used for the dot-product
        end = omp_get_wtime();
        // print_vector(y, csr_matrix->M);

        time_used = end - start;
        printf("Tempo seriale csr:%.16lf\n", time_used);

        compute_serial_performance(node, time_used, matrix->number_of_non_zeoroes_values);

        if (head == NULL)
        {
            head = (struct performance *)calloc(1, sizeof(struct performance));
            if (head == NULL)
            {
                printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
            }
            tail = (struct performance *)calloc(1, sizeof(struct performance));
            if (tail == NULL)
            {
                printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
            }

            head = node;
            tail = node;
        }
        else
        {
            tail->next_node = node;
            node->prev_node = tail;
            tail = node;
        }

        printf("Prestazioni Ottenute con il prodotto utilizzando il formato csr!\n");

        printf("\n\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
        printf("Time used for dot-product:      %.16lf\n", node->time_used);
        printf("FLOPS:                          %.16lf\n", node->flops);
        printf("MFLOPS:                         %.16lf\n", node->mflops);
        printf("GFLOPS:                         %.16lf\n\n", node->gflops);

        node = NULL;

        //
        // SERIAL EXECTUTION WITH HLL MATRIX FORMAT
        //

        node = (struct performance *)calloc(1, sizeof(struct performance));
        if (node == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        strcpy(node->matrix, csr_matrix->name);

        // Get statistics for the dot-product
        start = omp_get_wtime();
        // Start dot-product
        matvec_serial_hll(hll_matrix, x, z);
        // Get the time used for the dot-product
        end = omp_get_wtime();

        time_used = end - start;
        printf("Tempo seriale hll %.16lf\n", time_used);

        if (!compute_norm(y, z, csr_matrix->M, 1e-6))
        {
            printf("Errore nel controllo per %s\n", csr_matrix->name);
            sleep(3);
        }

        compute_serial_performance(node, time_used, matrix->number_of_non_zeoroes_values);
        node->non_zeroes_values = matrix->number_of_non_zeoroes_values;
        node->computation = SERIAL_HHL;

        if (head == NULL)
        {
            head = (struct performance *)calloc(1, sizeof(struct performance));
            if (head == NULL)
            {
                printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
            }
            tail = (struct performance *)calloc(1, sizeof(struct performance));
            if (tail == NULL)
            {
                printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
            }

            head = node;
            tail = node;
        }
        else
        {
            tail->next_node = node;
            node->prev_node = tail;
            tail = node;
        }

        printf("Prestazioni Ottenute con il prodotto utilizzando il formato hll in modalità seriale!\n");

        printf("\n\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
        printf("Time used for dot-product:      %.16lf\n", node->time_used);
        printf("FLOPS:                          %.16lf\n", node->flops);
        printf("MFLOPS:                         %.16lf\n", node->mflops);
        printf("GFLOPS:                         %.16lf\n\n", node->gflops);

        //
        // OpenMP CSR Matrix Format PARALLEL EXECUTION
        //

        node = NULL;

        printf("Prestazioni ottenute con OpenMP eseguendo il calcolo in parallelo!\n");
        // re_initialize_y_vector(csr_matrix->M, y);
        // re_initialize_y_vector(csr_matrix->M, z);

        node = (struct performance *)calloc(1, sizeof(struct performance));
        if (node == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        strcpy(node->matrix, matrix_filename);

        matvec_parallel_csr(csr_matrix, x, z, node, thread_numbers, head, tail, matrix->number_of_non_zeoroes_values);

        if (!compute_norm(y, z, csr_matrix->M, 1e-6))
        {
            printf("Errore nel controllo per %s dopo il csr parallelo\n", csr_matrix->name);
            sleep(3);
        }

        //
        // OpenMP HLL Matrix Format PARALLEL EXECUTION
        //

        node = (struct performance *)calloc(1, sizeof(struct performance));
        if (node == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        strcpy(node->matrix, matrix_filename);

        printf("Prestazioni Ottenute con il prodotto utilizzando il formato hll in modalità parallela!\n");
        re_initialize_y_vector(csr_matrix->M, z);

        matvec_parallel_hll(hll_matrix, x, z, node, thread_numbers, head, tail, matrix->number_of_non_zeoroes_values, y);

        // HERE STARTS CUDA IMPLEMENTATION
        // TODO MO DEVI FA LA PER LA CHIAMATA AL KERNEL CUDA

        invoke_kernel_1(hll_matrix, x, z, &time);

        printf("Time = %.16lf\n", time);

        node = NULL;
        destroy_HLL_matrix(hll_matrix);
        destroy_CSR_matrix(csr_matrix);
        (matrix);
        free(x);
        free(y);
        free(z);

        sleep(3);
    }

    closedir(dir);
    return 0;
}