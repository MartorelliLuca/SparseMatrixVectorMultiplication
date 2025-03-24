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
#include "../utils_header/computation_type.h"
#include "../../CUDA_include/cudacsr.h"

#define NUM_KERNELS 2

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

void invoke_cuda_csr_kernels(CSR_matrix *csr_matrix, double *x, double *z, double *effective_results, struct performance *head, struct performance *tail, struct performance *node)
{
    for (int i = 1; i <= NUM_KERNELS; i++) // TODO per ora ho messo 2 kernel, ma dobbiamo metterne 7
    {
        switch (i)
        {
        case 1:
            float time = invoke_kernel_csr_1(csr_matrix, x, z);
            break;

        case 2:
            float time = invoke_kernel_csr_2(csr_matrix, x, z);
            break;

        case 3:
            float time = invoke_kernel_csr_3(csr_matrix, x, z);
            break;
        case 4:
            float time = invoke_kernel_csr_4(csr_matrix, x, z);
            break;
        case 5:
            float time = invoke_kernel_csr_5(csr_matrix, x, z);
            break;
        case 6:
            float time = invoke_kernel_csr_6(csr_matrix, x, z);
            break;
        case 7:
            float time = invoke_kernel_csr_7(csr_matrix, x, z);
            break;
        }

        reset_node(node);
        compute_cuda_csr_kernel_results(node, (double)time, CUDA_HLL_KERNEL_0, 256, csr_matrix->non_zero_values);
        add_node_performance(head, tail, node);

        if (!compute_norm(effective_results, z, csr_matrix->M, 1e-6))
        {
            printf("Errore nel controllo per %s dopo il CUDA CSR kernel 0\n", csr_matrix->name);
            sleep(3);
        }
        sleep(2);
        print_cuda_csr_kernel_performance(node);
        sleep(2);
    }
}