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

void continue_execution(struct performance *node, struct performance **head, struct performance **tail, double *effective_results, double *z, CSR_matrix *csr_matrix, int i)
{
    add_node_performance(head, tail, node);

    if (!compute_norm(effective_results, z, csr_matrix->M, 1e-6))
    {
        printf("Errore nel controllo per %s dopo il CUDA CSR kernel %d\n", csr_matrix->name, i);
        sleep(2);
    }
    print_cuda_csr_kernel_performance(node);
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
    int num_thread_per_block[4] = {4, 8, 16, 32};
    for (int i = 1; i <= NUM_KERNELS; i++)
    {
        float time;
        switch (i)
        {
        case 1:
            for (int j = 1; j <= 4; j++)
            {
                // printf("STO A ENTRA IN INVOKE1, iterazione %d\n\n", j);
                re_initialize_y_vector(csr_matrix->M, z);
                time = invoke_kernel_csr_1(csr_matrix, x, z, num_thread_per_block[j - 1]);
                compute_cuda_csr_kernel_results(node, (double)time, CUDA_CSR_KERNEL_1, num_thread_per_block[j - 1], csr_matrix->non_zero_values);
                continue_execution(node, head, tail, effective_results, z, csr_matrix, i);
                node = NULL;
                node = reset_node();
            }
            break;

        case 2:
            for (int j = 1; j <= 4; j++)
            {
                // printf("STO A ENTRA IN INVOKE2, iterazione %d\n\n", j);
                re_initialize_y_vector(csr_matrix->M, z);
                time = invoke_kernel_csr_2(csr_matrix, x, z, num_thread_per_block[j - 1]);
                compute_cuda_csr_kernel_results(node, (double)time, CUDA_CSR_KERNEL_2, num_thread_per_block[j - 1], csr_matrix->non_zero_values);
                continue_execution(node, head, tail, effective_results, z, csr_matrix, i);
                node = NULL;
                node = reset_node();
            }
            break;

        case 3:
            for (int j = 1; j <= 4; j++)
            {
                // printf("STO A ENTRA IN INVOKE3, iterazione %d\n\n", j);
                re_initialize_y_vector(csr_matrix->M, z);
                time = invoke_kernel_csr_3(csr_matrix, x, z, num_thread_per_block[j - 1]);
                compute_cuda_csr_kernel_results(node, (double)time, CUDA_CSR_KERNEL_3, num_thread_per_block[j - 1], csr_matrix->non_zero_values);
                continue_execution(node, head, tail, effective_results, z, csr_matrix, i);
                node = NULL;
                node = reset_node();
            }
            break;

        case 4:
            for (int j = 1; j <= 4; j++)
            {
                re_initialize_y_vector(csr_matrix->M, z);
                // printf("STO A ENTRA IN INVOKE4, iterazione %d\n\n", j);
                time = invoke_kernel_csr_4(csr_matrix, x, z, num_thread_per_block[j - 1]);
                compute_cuda_csr_kernel_results(node, (double)time, CUDA_CSR_KERNEL_4, num_thread_per_block[j - 1], csr_matrix->non_zero_values);
                continue_execution(node, head, tail, effective_results, z, csr_matrix, i);
                node = NULL;
                node = reset_node();
            }
            break;
        }
    }
}
