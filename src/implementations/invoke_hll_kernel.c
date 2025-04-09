#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "../data_structures/hll_matrix.h"
#include "../data_structures/performance.h"
#include "../headers/hll_headers.h"
#include "../utils_header/utils.h"
#include "../utils_header/computation_type.h"
#include "../utils_header/initialization.h"
#include "../../CUDA_include/cudahll.h"

#define NUM_KERNEL 4
#define NUMBER_OF_ITERATION 5

void compute_cuda_hll_kernel_results(struct performance *node, double time, computation_time type, int threads_used, int non_zero_values)
{
    double flops = (2.0 * non_zero_values) / time;
    double mflops = flops / 1e6;
    double gflops = flops / 1e9;

    node->time_used = time;
    node->computation = type;
    node->number_of_threads_used = threads_used;
    node->flops = flops;
    node->gflops = gflops;
    node->mflops = mflops;
    node->non_zeroes_values = non_zero_values;
}

void invoke_cuda_hll_kernels(HLL_matrix *hll_matrix, double *x, double *z, double *effective_results, struct performance **head, struct performance **tail, struct performance *node)
{
    int threads_number[5] = {32, 64, 96, 128, 160};
    for (int i = 1; i <= NUM_KERNEL; i++)
    {
        float time_used, time = 0.0;
        switch (i)
        {
        case 1:
            for (int j = 0; j < 5; j++)
            {
                for (int k = 0; k < NUMBER_OF_ITERATION; k++)
                {
                    time_used = invoke_kernel_1(hll_matrix, x, z, threads_number[j]);
                    if (!compute_norm(z, effective_results, hll_matrix->M, 1e-6))
                    {
                        printf("Error in check for %s after CUDA hll kernel 1\n", hll_matrix->name);
                        sleep(3);
                    }
                    time += time_used;
                    re_initialize_y_vector(hll_matrix->M, z);
                }

                time = time / NUMBER_OF_ITERATION;
                compute_cuda_hll_kernel_results(node, (double)time, CUDA_HLL_KERNEL_1, threads_number[j], hll_matrix->data_num);
                add_node_performance(head, tail, node);
                print_cuda_hll_kernel_performance(node);
                node = NULL;
                node = reset_node();
            }
            time = 0.0;
            break;

        case 2:
            for (int j = 0; j < 5; j++)
            {
                for (int k = 0; k < NUMBER_OF_ITERATION; k++)
                {
                    time_used = invoke_kernel_2(hll_matrix, x, z, threads_number[j]);
                    if (!compute_norm(z, effective_results, hll_matrix->M, 1e-6))
                    {
                        printf("Error in check for %s after CUDA hll kernel 2\n", hll_matrix->name);
                        sleep(3);
                    }
                    time += time_used;
                    re_initialize_y_vector(hll_matrix->M, z);
                }

                time = time / NUMBER_OF_ITERATION;
                compute_cuda_hll_kernel_results(node, (double)time, CUDA_HLL_KERNEL_2, threads_number[j], hll_matrix->data_num);
                add_node_performance(head, tail, node);
                print_cuda_hll_kernel_performance(node);
                node = NULL;
                node = reset_node();
            }
            time = 0.0;
            break;

        case 3:
            for (int j = 0; j < 5; j++)
            {
                for (int k = 0; k < NUMBER_OF_ITERATION; k++)
                {
                    time_used = invoke_kernel_3(hll_matrix, x, z, threads_number[j]);
                    if (!compute_norm(z, effective_results, hll_matrix->M, 1e-6))
                    {
                        printf("Error in check for %s after CUDA hll kernel 3\n", hll_matrix->name);
                        sleep(3);
                    }
                    re_initialize_y_vector(hll_matrix->M, z);
                    time += time_used;
                }

                time = time / NUMBER_OF_ITERATION;
                compute_cuda_hll_kernel_results(node, (double)time, CUDA_HLL_KERNEL_3, threads_number[j], hll_matrix->data_num);
                add_node_performance(head, tail, node);
                print_cuda_hll_kernel_performance(node);
                node = NULL;
                node = reset_node();
            }
            time = 0.0;
            break;

        case 4:
            for (int j = 0; j < 5; j++)
            {
                for (int k = 0; k < NUMBER_OF_ITERATION; k++)
                {
                    time_used = invoke_kernel_4(hll_matrix, x, z, threads_number[j]);
                    if (!compute_norm(z, effective_results, hll_matrix->M, 1e-6))
                    {
                        printf("Error in check for %s after CUDA hll kernel 4\n", hll_matrix->name);
                        sleep(3);
                    }
                    time += time_used;
                    re_initialize_y_vector(hll_matrix->M, z);
                }

                time = time / NUMBER_OF_ITERATION;
                compute_cuda_hll_kernel_results(node, (double)time, CUDA_HLL_KERNEL_4, threads_number[j], hll_matrix->data_num);
                add_node_performance(head, tail, node);
                print_cuda_hll_kernel_performance(node);
                node = NULL;
                node = reset_node();
            }
            time = 0.0;
            break;
        }
    }
}