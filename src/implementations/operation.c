#include <ctype.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <bits/time.h>
#include <errno.h>
#include <unistd.h>
#include <immintrin.h>

#include "../data_structures/csr_matrix.h"
#include "../data_structures/hll_matrix.h"
#include "../data_structures/performance.h"
#include "../utils_header/utils.h"
#include "../utils_header/initialization.h"
#include "../utils_header/computation_type.h"

#define NUM_THREADS 39

void matvec(double **A, double *x, double *y, int M, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            y[i] += A[i][j] * x[j];
}

void matrix_partition(CSR_matrix *csr_matrix, int num_threads, int *start_row_indices)
{
    int *row_non_zero_count = (int *)malloc(csr_matrix->M * sizeof(int));
    int *workload_per_thread = (int *)calloc(num_threads, sizeof(int));
    double total_non_zero_elements = 0.0;

    for (int row = 0; row < csr_matrix->M; row++)
    {
        row_non_zero_count[row] = csr_matrix->IRP[row + 1] - csr_matrix->IRP[row];
        total_non_zero_elements += row_non_zero_count[row];
    }

    double target_workload_per_thread = total_non_zero_elements / num_threads;
    double current_thread_workload = 0.0;
    int current_thread_id = 0;
    start_row_indices[0] = 0;

    for (int row = 0; row < csr_matrix->M; row++)
    {
        current_thread_workload += row_non_zero_count[row];
        workload_per_thread[current_thread_id] += row_non_zero_count[row];

        if (current_thread_workload >= target_workload_per_thread && current_thread_id < num_threads - 1)
        {
            start_row_indices[current_thread_id + 1] = row + 1;
            current_thread_id++;
            current_thread_workload = 0.0;
        }
    }

    start_row_indices[num_threads] = csr_matrix->M;

    free(row_non_zero_count);
    free(workload_per_thread);
}

void product(CSR_matrix *csr_matrix, double *input_vector, double *output_vector, int num_threads, double *execution_time /*, double *times_vector*/)
{
    double start_time = omp_get_wtime();

#pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();

#pragma omp for schedule(dynamic, 32) // Bilanciamento del carico
        for (int row = 0; row < csr_matrix->M; row++)
        {
            double sum = 0.0;
            for (int idx = csr_matrix->IRP[row]; idx < csr_matrix->IRP[row + 1]; idx++)
            {
                sum += csr_matrix->AS[idx] * input_vector[csr_matrix->JA[idx]];
            }
            output_vector[row] = sum;
        }

        // printf("Thread %d: started: %.16lf, ended: %.16lf and took this time: %.16lf\n", thread_id, local_start_time, local_end_time, local_end_time - local_start_time);
        // times_vector[thread_id] = local_end_time - local_start_time;
    }

    double end_time = omp_get_wtime();
    *execution_time = end_time - start_time;
}

// First attempt to do matrix-vector dot product in CSR format
void matvec_serial_csr(CSR_matrix *csr_matrix, double *x, double *y)
{
    for (int i = 0; i < csr_matrix->M; i++)
    {
        for (int j = csr_matrix->IRP[i]; j < csr_matrix->IRP[i + 1]; j++)
        {
            y[i] += csr_matrix->AS[j] * x[csr_matrix->JA[j]];
        }
    }
}

void print_vector(double *y, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%.6lf ", y[i]);
    }
    printf("\n");
}

void compute_parallel_performance(struct performance *node, double *time_used, int new_non_zero_values, int num_threads)
{
    // Compute metrics
    double flops = 2.0 * new_non_zero_values / *time_used;
    double mflops = flops / 1e6;
    double gflops = flops / 1e9;

    node->number_of_threads_used = num_threads;
    node->flops = flops;
    node->mflops = mflops;
    node->gflops = gflops;
    node->time_used = *time_used;
    printf("node->time_used = %.16lf\n", node->time_used);
}

void matvec_parallel_csr(CSR_matrix *csr_matrix, double *x, double *y, struct performance *node, int *thread_numbers, struct performance *head, struct performance *tail, int new_non_zero_values, double *effective_results)
{

    double time_used, time_used2;
    for (int index = 0; index < NUM_THREADS; index++)
    {
        int num_threads = thread_numbers[index];

        // partiziona il lavoro tra i thread
        int *first_row;

        first_row = (int *)calloc(num_threads + 1, sizeof(int)); // Allocare num_threads + 1 elementi
        if (!first_row)
        {
            perror("Errore in calloc per first_row");
            exit(EXIT_FAILURE);
        }

        double *times_vector = (double *)calloc(num_threads, sizeof(double));
        if (!times_vector)
        {
            perror("Errore in calloc per times_vector");
            exit(EXIT_FAILURE);
        }

        matrix_partition(csr_matrix, num_threads, first_row);

        product(csr_matrix, x, y, num_threads, &time_used);
        compute_parallel_performance(node, &time_used, new_non_zero_values, num_threads);
        printf("Tempo utilizzato per l'esecuzione con %d con csr parallelo = %.lf\n", num_threads, node->time_used);

        if (!compute_norm(y, effective_results, csr_matrix->M, 1e-6))
        {
            printf("Errore nel controllo per %s dopo il csr parallelo\n", csr_matrix->name);
            sleep(1);
        }

        node->non_zeroes_values = new_non_zero_values;
        node->computation = PARALLEL_OPEN_MP_CSR;

        add_node_performance(head, tail, node);

        print_parallel_csr_result(node);

        // fai il prodotto scalare tra la riga i-esima e il vettore x
    }
}

static inline int num_of_rows(HLL_matrix *hll, int h)
{
    if (hll->hacks_num - 1 == h && hll->M % hll->hack_size)
    {
        return hll->M % hll->hack_size;
    }
    return hll->hack_size;
}

// Matrix-vector serial dot produt in HLL format
void matvec_serial_hll(HLL_matrix *hll_matrix, double *x, double *y)
{
    int rows = num_of_rows(hll_matrix, 0);
    for (int h = 0; h < hll_matrix->hacks_num; ++h)
    {
        for (int r = 0; r < num_of_rows(hll_matrix, h); ++r)
        {
            double sum = 0.0;
            for (int j = 0; j < hll_matrix->max_nzr[h]; ++j)
            {
                int k = hll_matrix->offsets[h] + r * hll_matrix->max_nzr[h] + j;
                sum += hll_matrix->data[k] * x[hll_matrix->col_index[k]];
            }
            y[rows * h + r] = sum;
        }
    }
}

void hll_parallel_product(HLL_matrix *restrict hll_matrix, double *restrict x, double *restrict y,
                          int num_threads, double *time_used)
{
    const int rows = num_of_rows(hll_matrix, 0);
    double start = omp_get_wtime();

#pragma omp parallel for schedule(guided) num_threads(num_threads) default(none) \
    shared(hll_matrix, x, y, rows)
    for (int h = 0; h < hll_matrix->hacks_num; ++h)
    {
        const int rows_in_h = num_of_rows(hll_matrix, h);
        const int max_nzr_h = hll_matrix->max_nzr[h];
        const int offset_h = hll_matrix->offsets[h];

        for (int r = 0; r < rows_in_h; ++r)
        {
            double sum = 0.0;
            double *data_ptr = &hll_matrix->data[offset_h + r * max_nzr_h];
            int *col_ptr = &hll_matrix->col_index[offset_h + r * max_nzr_h];

#pragma omp simd reduction(+ : sum)
            for (int j = 0; j < max_nzr_h; ++j)
            {
                __builtin_prefetch(&data_ptr[j + 4], 0, 1); // Prefetch della riga successiva
                __builtin_prefetch(&col_ptr[j + 4], 0, 1);

                sum += data_ptr[j] * x[col_ptr[j]];
            }

            y[rows * h + r] = sum;
        }
    }

    double end = omp_get_wtime();
    *time_used = end - start;
}

void compute_hll_parallel_performance(struct performance *node, int new_non_zero_values, double time_used, int num_threads)
{
    node->non_zeroes_values = new_non_zero_values;
    node->computation = PARALLEL_OPEN_MP_HLL;

    printf("Time used for dot-product: %.16lf\n", time_used);

    double flops = 2.0 * new_non_zero_values / time_used;
    double mflops = flops / 1e6;
    double gflops = flops / 1e9;

    node->number_of_threads_used = num_threads;
    node->flops = flops;
    node->mflops = mflops;
    node->gflops = gflops;
    node->time_used = time_used;

    printf("Tempo utilizzato per hll parallelo utilizzando %d thread = %.16lf\n", num_threads, node->time_used);
}

// Matrix-vector parallel dot product in HLL format
void matvec_parallel_hll(HLL_matrix *hll_matrix, double *x, double *y, struct performance *node, int *thread_numbers, struct performance *head, struct performance *tail, int new_non_zero_values, double *effective_results)
{
    double time_used, start, end;
    for (int index = 0; index < NUM_THREADS; index++)
    {
        int num_threads = thread_numbers[index];

        double *times_vector = (double *)calloc(num_threads, sizeof(double));
        if (!times_vector)
        {
            perror("Errore in calloc per times_vector");
            exit(EXIT_FAILURE);
        }

        hll_parallel_product(hll_matrix, x, y, num_threads, &time_used);

        compute_hll_parallel_performance(node, new_non_zero_values, time_used, num_threads);

        add_node_performance(head, tail, node);

        print_parallel_hll_result(node);

        if (!compute_norm(y, effective_results, hll_matrix->M, 1e-6))
        {
            printf("Errore nel controllo per %s dopo il hll parallelo\n", hll_matrix->name);
            sleep(3);
        }

        re_initialize_y_vector(hll_matrix->M, y);
    }
}
