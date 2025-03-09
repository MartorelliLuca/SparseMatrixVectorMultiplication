#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <time.h>

#include "utils_header/mmio.h"
#include "data_structures/hll_matrix.h"
#include "data_structures/csr_matix.h"
#include "data_structures/performance.h"
#include "headers/matrix.h"
#include "headers/csr_headers.h"
#include "headers/hll_headers.h"
#include "headers/operation.h"
#include "utils_header/initialization.h"

#define MATRIX_DIR = "/matrici"

int main()
{
    int thread_numbers[6] = {2, 4, 8, 16, 32, 40};
    int matrix_counter = 0; // Number of matrices processed

    int file_type[2];
    // file_type[0] -> is_sparse_matrix
    // file_type[1] -> is_array_file

    // Variables to collect statistics
    struct performance *head = NULL, *tail = NULL, *node = NULL;
    struct timespec start, end;
    double time_used;
    double flops, mflops, gflops;
    int new_non_zero_values;

    const char *dir_name = "matrici";
    char matrix_filename[256]; // Matrix filename
    char matrix_fullpath[256]; // Buffer to full path of the matrix file to open
    DIR *dir;
    struct dirent *entry;
    FILE *matrix_file;

    CSR_matrix csr_matrix;
    HLL_matrix ellpack_matrix;

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
        if (entry->d_name[0] == '.')
            continue;

        matrix_counter++;

        strcpy(matrix_filename, &entry->d_name);

        printf("Processing %s matrix\n", matrix_filename);

        // For now, this matrix give error banner read but I don't know why. For now, skip
        if (strcmp(matrix_filename, "amazon0302.mtx") == 0 || strcmp(matrix_filename, "roadNet-PA.mtx") == 0)
            continue;

        matrix_file = get_matrix_file(dir_name, matrix_filename, file_type);

        // Get matrix from matrix market format in csr format
        read_CSR_matrix(matrix_file, &csr_matrix, file_type);

        strcpy(csr_matrix.name, matrix_filename);

        // Initialize x and y vector
        x = initialize_x_vector(csr_matrix.M);
        y = initialize_y_vector(csr_matrix.M);

        //
        // SERIAL EXECUTION WITH CSR MATRIX FORMAT
        //
        // Get statistics for the dot-product
        clock_gettime(CLOCK_MONOTONIC, &start);
        // Start dot-product
        matvec_csr(&csr_matrix, x, y);
        // Get the time used for the dot-product
        clock_gettime(CLOCK_MONOTONIC, &end);

        // Get time used
        time_used = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        // Compute metrics
        flops = 2.0 * csr_matrix.NZ / time_used;
        mflops = flops / 1e6;
        gflops = flops / 1e9;

        // Allocating the node to save performance
        node = (struct performance *)calloc(1, sizeof(struct performance));
        if (node == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
        }

        strcpy(node->matrix, csr_matrix.name);
        node->number_of_threads_used = 1;
        node->flops = flops;
        node->mflops = mflops;
        node->gflops = gflops;
        node->time_used = time_used;

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

        printf("Prestazioni Ottenute con il prodotto con csr!\n");

        printf("\n\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
        printf("Time used for dot-product:      %.16lf\n", node->time_used);
        printf("FLOPS:                          %.16lf\n", node->flops);
        printf("MFLOPS:                         %.16lf\n", node->mflops);
        printf("GFLOPS:                         %.16lf\n\n", node->gflops);

        node = NULL;

        printf("\n\n Esecuzione con la divisione in chunck!\n");

        // SERIAL EXECUTION WITH MATRIX ALL IN MEMORY
        // Get statistics for the dot-product
        clock_gettime(CLOCK_MONOTONIC, &start);
        // Start dot-product
        // Recreate full matrix in memory and then perform the product
        new_non_zero_values = (&csr_matrix, x, y, 5000);
        // Get the time used for the dot-product
        clock_gettime(CLOCK_MONOTONIC, &end);

        // Compute metrics
        flops = 2.0 * new_non_zero_values / time_used;
        mflops = flops / 1e6;
        gflops = flops / 1e9;

        // Allocating the node to save performance
        node = (struct performance *)calloc(1, sizeof(struct performance));
        if (node == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
        }

        strcpy(node->matrix, csr_matrix.name);
        node->number_of_threads_used = 1;
        node->flops = flops;
        node->mflops = mflops;
        node->gflops = gflops;
        node->time_used = time_used;

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

        printf("\n\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
        printf("Time used for dot-product:      %.16lf\n", node->time_used);
        printf("FLOPS:                          %.16lf\n", node->flops);
        printf("MFLOPS:                         %.16lf\n", node->mflops);
        printf("GFLOPS:                         %.16lf\n\n", node->gflops);

        node = NULL;

        //
        // OpenMP EXECUTION
        //

        printf("Prestazioni ottenute con OpenMP!\n");

        for (int index = 0; index < 6; index++)
        {
            int num_threads = thread_numbers[index];

            // Set number of threads to perform dot product executions
            omp_set_num_threads(num_threads);

            // Get statistics for the dot-product
            clock_gettime(CLOCK_MONOTONIC, &start);
            // Start dot-product
            matvec_csr(&csr_matrix, x, y);
            // Get the time used for the dot-product
            clock_gettime(CLOCK_MONOTONIC, &end);

            // Get time used
            time_used = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

            // Compute metrics
            flops = 2.0 * csr_matrix.NZ / time_used;
            mflops = flops / 1e6;
            gflops = flops / 1e9;

            // Allocating the node to save performance
            node = (struct performance *)calloc(1, sizeof(struct performance));
            if (node == NULL)
            {
                printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
            }

            strcpy(node->matrix, matrix_filename);
            node->number_of_threads_used = num_threads;
            node->flops = flops;
            node->mflops = mflops;
            node->gflops = gflops;
            node->time_used = time_used;

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

            printf("\n\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
            printf("Time used for dot-product:      %.16lf\n", node->time_used);
            printf("FLOPS:                          %.16lf\n", node->flops);
            printf("MFLOPS:                         %.16lf\n", node->mflops);
            printf("GFLOPS:                         %.16lf\n\n", node->gflops);
        }

        node = NULL;

        // // Read Values
        // while (head != NULL)
        // {
        //     node = head;
        //     printf("\n\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
        //     printf("Time used for dot-product:      %.16lf\n", node->time_used);
        //     printf("FLOPS:                          %.16lf\n", node->flops);
        //     printf("MFLOPS:                         %.16lf\n", node->mflops);
        //     printf("GFLOPS:                         %.16lf\n\n", node->gflops);
        //     head = head->next_node;
        // }

        fclose(matrix_file);

        sleep(3);
    }

    closedir(dir);
    return 0;
}