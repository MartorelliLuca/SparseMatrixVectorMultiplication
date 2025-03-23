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
#include "../../CUDA_include/cudahll.h"

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

void invoke_cuda_hll_kernels(HLL_matrix *hll_matrix, double *x, double *z, double *effective_results, struct performance *head, struct performance *tail, struct performance *node)
{
    float time = invoke_kernel_1(hll_matrix, x, z);

    reset_node(node);

    compute_cuda_hll_kernel_results(node, (double)time, CUDA_HLL_KERNEL_0, 256, hll_matrix->data_num);

    add_node_performance(head, tail, node);

    if (!compute_norm(effective_results, z, hll_matrix->M, 1e-6))
    {
        printf("Errore nel controllo per %s dopo il CUDA hll kernel 0\n", hll_matrix->name);
        sleep(3);
    }

    print_cuda_hll_kernel_performance(node);
}