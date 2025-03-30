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
#include "headers/invoke_hll_kernel.h"
#include "headers/invoke_csr_kernel.h"
#include "utils_header/initialization.h"
#include "utils_header/utils.h"
#include "utils_header/computation_type.h"
#include "../CUDA_include/cudahll.h"
#include "../CUDA_include/cudacsr.h"

#define MATRIX_DIR = "../matrici"

int main()
{
    int matrix_counter = 0; // Number of matrices processed

    // Variables to collect statistics
    struct performance *head = NULL, *tail = NULL, *node = NULL;
    double time_used, start, end;

    const char *dir_name = "../matrici";
    // Matrix filename
    char matrix_filename[256];
    DIR *dir;
    struct dirent *entry;
    FILE *matrix_file;
    matrix_format *matrix = NULL;

    CSR_matrix *csr_matrix = NULL;
    HLL_matrix *hll_matrix = NULL;

    // Number of threads used for calculations
    int *thread_numbers = initialize_threads_number();

    double *x, *y, *z;

    // Open dir matrix to take the tests matrix
    dir = opendir(dir_name);
    if (dir == NULL)
    {
        printf("Error occour while opening the matrix directory!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    // Take one test matrix every time and transform it to csr and hll format
    // Compute metrics for csr and hll format in serial, parallel with OpenMP and
    // with CUDA library
    while ((entry = readdir(dir)) != NULL)
    {

        csr_matrix = get_csr_matrix();

        hll_matrix = get_hll_matrix();

        matrix = get_matrix_format_matrix();

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

        mtx_cleanup(matrix);

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
        add_node_performance(&head, &tail, node);
        print_serial_csr_result(node);

        print_list(head);

        //
        // SERIAL EXECTUTION WITH HLL MATRIX FORMAT
        //

        node = NULL;
        node = reset_node();

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

        add_node_performance(&head, &tail, node);

        re_initialize_y_vector(csr_matrix->M, z);

        print_serial_hll_result(node);

        //
        // OpenMP CSR Matrix Format PARALLEL EXECUTION
        //

        // printf("PRINTO LA LISTA DOPO HLL SERIALE\n");
        // print_list(head);

        printf("Prestazioni ottenute con OpenMP eseguendo il calcolo in parallelo!\n");

        node = NULL;
        node = reset_node();
        strcpy(node->matrix, matrix_filename);

        re_initialize_y_vector(csr_matrix->M, z);

        matvec_parallel_csr(csr_matrix, x, z, node, thread_numbers, &head, &tail, matrix->number_of_non_zeoroes_values, y);

        // printf("PRINTO LA LISTA DOPO CSR PARALLELO\n");
        // print_list(head);

        sleep(3);

        //
        // OpenMP HLL Matrix Format PARALLEL EXECUTION
        //

        printf("Prestazioni Ottenute con il prodotto utilizzando il formato hll in modalità parallela!\n");

        node = NULL;
        node = reset_node();
        strcpy(node->matrix, matrix_filename);

        re_initialize_y_vector(csr_matrix->M, z);

        matvec_parallel_hll(hll_matrix, x, z, node, thread_numbers, &head, &tail, matrix->number_of_non_zeoroes_values, y);

        // printf("PRINTO LA LISTA DOPO HLL PARALLELO\n");
        // print_list(head);

        sleep(3);

        node = NULL;
        node = reset_node();

        // HERE STARTS CUDA IMPLEMENTATION
        invoke_cuda_csr_kernels(csr_matrix, x, z, y, &head, &tail, node);

        node = NULL;
        node = reset_node();
        invoke_cuda_hll_kernels(hll_matrix, x, z, y, &head, &tail, node);

        printf("\n\nSTAMPO LA LISTA PER %s\n\n", csr_matrix->name);
        print_list(head);

        node = NULL;
        destroy_HLL_matrix(hll_matrix);
        destroy_CSR_matrix(csr_matrix);
        free(x);
        free(y);
        free(z);

        sleep(3);
    }

    closedir(dir);
    return 0;
}