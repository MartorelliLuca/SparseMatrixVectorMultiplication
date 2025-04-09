#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "../data_structures/csr_matrix.h"
#include "../data_structures/performance.h"
#include "../headers/csr_headers.h"
#include "../utils_header/utils.h"
#include "../utils_header/initialization.h"
#include "../utils_header/computation_type.h"
#include "../../CUDA_include/cudacsr.h"

#define NUM_KERNELS 4
#define NUM_THREADS 5
#define NUMBER_OF_COMPUTATION 5

void continue_execution(struct performance *node, struct performance **head, struct performance **tail, double *effective_results, double *z, CSR_matrix *csr_matrix, int i)
{
    if (!compute_norm(z, effective_results, csr_matrix->M, 1e-6))
    {
        printf("Error in check for %s after CUDA CSR kernel %d\n", csr_matrix->name, i);
        sleep(2);
    }
}

void compute_cuda_csr_kernel_results(struct performance *node, double time, computation_time type, int threads_used, int non_zero_values)
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

void invoke_cuda_csr_kernels(CSR_matrix *csr_matrix, double *x, double *z, double *effective_results, struct performance **head, struct performance **tail, struct performance *node)
{
    int num_thread_per_block[NUM_THREADS] = {1, 2, 3, 4, 5};
    for (int i = 1; i <= NUM_KERNELS; i++)
    {
        float time_used, time = 0.0;
        switch (i)
        {
        case 1:
            for (int j = 1; j <= NUM_THREADS; j++)
            {
                for (int k = 0; k < NUMBER_OF_COMPUTATION; k++)
                {
                    time_used = invoke_kernel_csr_1(csr_matrix, x, z, num_thread_per_block[j - 1]);
                    continue_execution(node, head, tail, effective_results, z, csr_matrix, i);
                    time += time_used;
                    re_initialize_y_vector(csr_matrix->M, z);
                }
                time = time / NUMBER_OF_COMPUTATION;
                compute_cuda_csr_kernel_results(node, (double)time, CUDA_CSR_KERNEL_1, num_thread_per_block[j - 1], csr_matrix->non_zero_values);
                add_node_performance(head, tail, node);
                print_cuda_csr_kernel_performance(node);
                node = NULL;
                node = reset_node();
            }
            time = 0.0;
            break;

        case 2:
            for (int j = 1; j <= NUM_THREADS; j++)
            {
                for (int k = 0; k < NUMBER_OF_COMPUTATION; k++)
                {
                    time_used = invoke_kernel_csr_2(csr_matrix, x, z, num_thread_per_block[j - 1]);
                    continue_execution(node, head, tail, effective_results, z, csr_matrix, i);
                    time += time_used;
                    re_initialize_y_vector(csr_matrix->M, z);
                }

                time = time / NUMBER_OF_COMPUTATION;
                compute_cuda_csr_kernel_results(node, (double)time, CUDA_CSR_KERNEL_2, num_thread_per_block[j - 1], csr_matrix->non_zero_values);
                add_node_performance(head, tail, node);
                print_cuda_csr_kernel_performance(node);
                node = NULL;
                node = reset_node();
            }
            time = 0.0;
            break;

        case 3:
            for (int j = 1; j <= NUM_THREADS; j++)
            {
                for (int k = 0; k < NUMBER_OF_COMPUTATION; k++)
                {
                    time_used = invoke_kernel_csr_3(csr_matrix, x, z, num_thread_per_block[j - 1]);
                    continue_execution(node, head, tail, effective_results, z, csr_matrix, i);
                    time += time_used;
                    re_initialize_y_vector(csr_matrix->M, z);
                }
                time = time / NUMBER_OF_COMPUTATION;
                compute_cuda_csr_kernel_results(node, (double)time, CUDA_CSR_KERNEL_3, num_thread_per_block[j - 1], csr_matrix->non_zero_values);
                add_node_performance(head, tail, node);
                print_cuda_csr_kernel_performance(node);
                node = NULL;
                node = reset_node();
            }
            time = 0.0;
            break;

        case 4:
            for (int j = 1; j <= NUM_THREADS; j++)
            {
                for (int k = 0; k < NUMBER_OF_COMPUTATION; k++)
                {
                    time_used = invoke_kernel_csr_4(csr_matrix, x, z, num_thread_per_block[j - 1]);
                    continue_execution(node, head, tail, effective_results, z, csr_matrix, i);
                    time += time_used;
                    re_initialize_y_vector(csr_matrix->M, z);
                }

                time = time / NUMBER_OF_COMPUTATION;
                compute_cuda_csr_kernel_results(node, (double)time, CUDA_CSR_KERNEL_4, num_thread_per_block[j - 1], csr_matrix->non_zero_values);
                add_node_performance(head, tail, node);
                print_cuda_csr_kernel_performance(node);
                node = NULL;
                node = reset_node();
            }
            time = 0.0;
            break;
        }
    }
}