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
#include "data_structures/performance.h"
#include "headers/csr_headers.h"
#include "headers/ellpack_headers.h"
#include "headers/operation.h"
#include "utils_header/initialization.h"

int main()
{

    int matrix_counter = 0; // Number of matrices processed

    const char *dir_name = "matrici";
    char matrix_filename[256]; // Matrix filename
    char matrix_fullpath[256]; // Buffer to full path of the matrix file to open
    DIR *dir;
    struct dirent *entry;
    FILE *matrix_file;

    CSR_matrix csr_matrix;
    ELLPACK_matrix ellpack_matrix;

    double *x;
    double *y;

    // Variables to collect statistics
    struct performance *head = NULL, *tail = NULL, *node = NULL;
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

        matrix_file = get_matrix_file(dir_name, matrix_filename);

        // Get matrix from matrix market format in csr format
        read_CSR_matrix(matrix_file, &csr_matrix);

        // Initialize x and y vector
        x = initialize_x_vector(csr_matrix.N);
        y = initialize_y_vector(csr_matrix.M);

        for (int num_threads = 1; num_threads <= 40; num_threads++)
        {
            omp_set_num_threads(num_threads);
            // Get statistics for the dot-product
            clock_gettime(CLOCK_MONOTONIC, &start);

            // Start dot-product
            matvec_csr(&csr_matrix, x, y);

            // Get the time used for the dot-product
            clock_gettime(CLOCK_MONOTONIC, &end);
            time_used = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

            flops = 2.0 * csr_matrix.NZ / time_used;
            mflops = flops / 1e6;
            gflops = flops / 1e9;

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
        }

        while (head != NULL)
        {
            node = head;
            printf("\n\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
            printf("Time used for dot-product:      %.16lf\n", node->time_used);
            printf("FLOPS:                          %.16lf\n", node->flops);
            printf("MFLOPS:                         %.16lf\n", node->mflops);
            printf("GFLOPS:                         %.16lf\n\n", node->gflops);
            head = head->next_node;
        }

        fclose(matrix_file);
    }

    closedir(dir);
    return 0;
}