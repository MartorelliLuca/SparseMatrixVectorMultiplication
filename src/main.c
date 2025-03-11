#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <time.h>

#include "utils_header/mmio.h"
#include "data_structures/hll_matrix.h"
#include "data_structures/csr_matrix.h"
#include "data_structures/performance.h"
#include "headers/matrix.h"
#include "headers/csr_headers.h"
#include "headers/hll_headers.h"
#include "headers/operation.h"
#include "utils_header/initialization.h"

#define MATRIX_DIR = "../matrici"

int main()
{
    int thread_numbers[6] = {2, 4, 8, 16, 32, 40};
    int matrix_counter = 0; // Number of matrices processed

    // Variables to collect statistics
    struct performance *head = NULL, *tail = NULL, *node = NULL;
    double time_used, start, end;
    double flops, mflops, gflops;
    int new_non_zero_values;
    int symmetric = 0;

    const char *dir_name = "../matrici";
    char matrix_filename[256]; // Matrix filename
    char matrix_fullpath[256]; // Buffer to full path of the matrix file to open
    DIR *dir;
    struct dirent *entry;
    FILE *matrix_file;
    matrix_format *matrix = NULL;

    CSR_matrix *csr_matrix = NULL;
    HLL_matrix *hll_matrix = NULL;

    double *x;
    double *y;

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

        if (entry->d_name[0] == '.')
            continue;

        matrix_counter++;

        strcpy(matrix_filename, &entry->d_name);

        printf("Processing %s matrix\n", matrix_filename);

        // For now, this matrix give error banner read but I don't know why. For now, skip
        if (strcmp(matrix_filename, "amazon0302.mtx") == 0 || strcmp(matrix_filename, "roadNet-PA.mtx") == 0 || strcmp(matrix_filename, "cop20k_A.mtx") == 0)
            continue;

        matrix = (matrix_format *)calloc(1, sizeof(matrix_format));
        if (matrix == NULL)
        {
            printf("Error occour in malloc for matrix format varibale!\nError Code:%d\n", errno);
            exit(EXIT_FAILURE);
        }

        matrix_file = get_matrix_file(dir_name, matrix_filename);

        // Get matrix from matrix market format in csr format
        read_CSR_matrix(matrix_file, csr_matrix);

        // Get matrix from matrix market format in hll format
        read_HLL_matrix(csr_matrix, hll_matrix);

        strcpy(csr_matrix->name, matrix_filename);

        // Initialize x and y vector
        x = initialize_x_vector(csr_matrix->M);
        y = initialize_y_vector(csr_matrix->M);

        new_non_zero_values = get_real_non_zero_values_count(csr_matrix);

        // Debug Print
        // printf("New non-zero values:        %d.\n", new_non_zero_values);
        // printf("Non zero values reaeds:     %d.\n", csr_matrix.NZ);

        // SERIAL EXECUTION WITH CSR MATRIX FORMAT
        // Allocating the node to save performance
        node = (struct performance *)calloc(1, sizeof(struct performance));
        if (node == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
        }
        strcpy(node->matrix, csr_matrix->name);
        // Get statistics for the dot-product
        start = omp_get_wtime();
        // Start dot-product
        matvec_serial_csr(csr_matrix, x, y);
        // Get the time used for the dot-product
        end = omp_get_wtime();

        time_used = end - start;

        compute_serial_performance(node, time_used, new_non_zero_values);

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

        // SERIAL EXECTUTION WITH HLL MATRIX FORMAT
        //

        node = (struct performance *)calloc(1, sizeof(struct performance));
        if (node == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        strcpy(node->matrix, csr_matrix->name);

        re_initialize_y_vector(hll_matrix->M, y);

        // Get statistics for the dot-product
        start = omp_get_wtime();
        // Start dot-product
        matvec_serial_hll(hll_matrix, x, y);
        // Get the time used for the dot-product
        end = omp_get_wtime();

        time_used = end - start;

        new_non_zero_values = get_real_non_zero_values_count(csr_matrix);

        compute_serial_performance(node, time_used, new_non_zero_values);

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

        printf("Prestazioni Ottenute con il prodotto utilizzando il formato hll!\n");

        printf("\n\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
        printf("Time used for dot-product:      %.16lf\n", node->time_used);
        printf("FLOPS:                          %.16lf\n", node->flops);
        printf("MFLOPS:                         %.16lf\n", node->mflops);
        printf("GFLOPS:                         %.16lf\n\n", node->gflops);

        //
        // OpenMP EXECUTION
        //

        node = NULL;

        printf("Prestazioni ottenute con OpenMP!\n");
        re_initialize_y_vector(csr_matrix->M, y);
        node = (struct performance *)calloc(1, sizeof(struct performance));
        if (node == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
            exit(EXIT_FAILURE);
        }

        strcpy(node->matrix, matrix_filename);

        matvec_parallel_csr(csr_matrix, x, y, node, thread_numbers, head, tail, new_non_zero_values);

        node = NULL;

        sleep(10);
    }

    closedir(dir);
    return 0;
}