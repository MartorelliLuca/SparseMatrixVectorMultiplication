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
#define TOTAL_FILES 30

int main()
{
    int matrix_counter = 0;

    struct performance *head = NULL, *tail = NULL, *node = NULL;
    double time_used, start, end;

    const char *dir_name = "../matrici";

    char matrix_filename[256];
    DIR *dir;
    struct dirent *entry;
    FILE *matrix_file;
    matrix_format *matrix = NULL;

    CSR_matrix *csr_matrix = NULL;
    HLL_matrix *hll_matrix = NULL;

    int *thread_numbers = initialize_threads_number();
    int processed_matices = 0;
    int old_processed_matrices = -1;

    double *x, *y, *z;

    // Open dir matrix to take the tests matrix
    dir = opendir(dir_name);
    if (dir == NULL)
    {
        printf("Error occour while opening the matrix directory!\nError code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    print_title();
    puts("");
    puts("");

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
        matrix_file = get_matrix_file(dir_name, matrix_filename);

        if (read_matrix(matrix_file, matrix) == 1)
        {
            printf("Error in read matrix!\n");
            exit(EXIT_FAILURE);
        }

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
            exit(EXIT_FAILURE);
        }

        strcpy(node->matrix, csr_matrix->name);
        node->non_zeroes_values = matrix->number_of_non_zeoroes_values;
        node->computation = SERIAL_CSR;

        // Get statistics for the dot-product
        start = omp_get_wtime();
        matvec_serial_csr(csr_matrix, x, y);
        end = omp_get_wtime();
        time_used = end - start;

        compute_serial_performance(node, time_used, matrix->number_of_non_zeoroes_values);
        add_node_performance(&head, &tail, node);

        //
        // SERIAL EXECTUTION WITH HLL MATRIX FORMAT
        //

        node = NULL;
        node = reset_node();

        strcpy(node->matrix, csr_matrix->name);

        // Get statistics for the dot-product
        start = omp_get_wtime();
        matvec_serial_hll(hll_matrix, x, z);
        end = omp_get_wtime();

        time_used = end - start;

        if (!compute_norm(y, z, csr_matrix->M, 1e-6))
        {
            printf("Errore nel controllo per %s\n", csr_matrix->name);
        }

        compute_serial_performance(node, time_used, matrix->number_of_non_zeoroes_values);
        node->non_zeroes_values = matrix->number_of_non_zeoroes_values;
        node->computation = SERIAL_HHL;

        add_node_performance(&head, &tail, node);

        re_initialize_y_vector(csr_matrix->M, z);

        //
        // OpenMP CSR Matrix Format PARALLEL EXECUTION
        //

        node = NULL;
        node = reset_node();
        strcpy(node->matrix, matrix_filename);

        re_initialize_y_vector(csr_matrix->M, z);

        matvec_parallel_csr(csr_matrix, x, z, node, thread_numbers, &head, &tail, matrix->number_of_non_zeoroes_values, y);

        //
        // OpenMP HLL Matrix Format PARALLEL EXECUTION
        //

        node = NULL;
        node = reset_node();
        strcpy(node->matrix, matrix_filename);

        re_initialize_y_vector(csr_matrix->M, z);
        matvec_parallel_hll(hll_matrix, x, z, node, thread_numbers, &head, &tail, matrix->number_of_non_zeoroes_values, y);

        node = NULL;
        node = reset_node();

        // HERE STARTS CUDA IMPLEMENTATION
        invoke_cuda_csr_kernels(csr_matrix, x, z, y, &head, &tail, node);

        node = NULL;
        node = reset_node();
        invoke_cuda_hll_kernels(hll_matrix, x, z, y, &head, &tail, node);

        save_performance_to_csv(head);
        free_performance_list(&head);
        node = NULL;
        destroy_HLL_matrix(hll_matrix);
        destroy_CSR_matrix(csr_matrix);
        free(x);
        free(y);
        free(z);

        processed_matices++;
        print_progress_bar("Progress", processed_matices, TOTAL_FILES, old_processed_matrices);
        old_processed_matrices = processed_matices;
    }

    puts("");
    puts("");
    closedir(dir);
    return 0;
}