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
#define COMPUTATION 5

void matvec(double **A, double *x, double *y, int M, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            y[i] += A[i][j] * x[j];
}

//    (int *actual_num_threads). Ci scriverai dentro i thread realmente usati:
int *matrix_partition(CSR_matrix *csr_matrix, int num_threads, int *actual_num_threads)
{
    int *row_non_zero_count = malloc(csr_matrix->M * sizeof(int));
    double total_non_zero_elements = 0.0;

    // Conta i non-zero per riga
    for (int row = 0; row < csr_matrix->M; row++)
    {
        row_non_zero_count[row] = csr_matrix->IRP[row + 1] - csr_matrix->IRP[row];
        total_non_zero_elements += row_non_zero_count[row];
    }

    // Calcolo del carico target
    double target_workload_per_thread = total_non_zero_elements / num_threads;

    // Alloca +1 per salvare anche la riga finale (M)
    int *temp_start_row_indices = malloc((num_threads + 1) * sizeof(int));

    double current_thread_workload = 0.0;
    int current_thread_id = 0;

    temp_start_row_indices[0] = 0;

    for (int row = 0; row < csr_matrix->M; row++)
    {
        current_thread_workload += row_non_zero_count[row];

        if (current_thread_workload >= target_workload_per_thread &&
            current_thread_id < num_threads - 1)
        {
            current_thread_id++;
            temp_start_row_indices[current_thread_id] = row + 1;
            current_thread_workload = 0.0;
        }
    }

    // current_thread_id + 1 = numero di thread effettivamente usati
    temp_start_row_indices[current_thread_id + 1] = csr_matrix->M;

    // Esegui eventuale realloc se hai usato meno thread
    int used_threads = current_thread_id + 1;
    int new_size = used_threads + 1;
    temp_start_row_indices = realloc(temp_start_row_indices, new_size * sizeof(int));
    if (!temp_start_row_indices)
    {
        fprintf(stderr, "ERRORE REALLOC temp_start_row_indices\n");
        exit(EXIT_FAILURE);
    }

    // 2. Salvi il numero effettivo di thread nel parametro passato dal main
    *actual_num_threads = used_threads;

    free(row_non_zero_count);

    return temp_start_row_indices;
}

void product(CSR_matrix *csr_matrix, double *input_vector, double *output_vector, int num_threads, double *execution_time, int *first_row)
{
    double start_time = omp_get_wtime();

#pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        for (int row = first_row[thread_id]; row < first_row[thread_id + 1] - 1; row++)
        {
            double sum = 0.0;
            for (int idx = csr_matrix->IRP[row]; idx < csr_matrix->IRP[row + 1]; idx++)
            {
                sum += csr_matrix->AS[idx] * input_vector[csr_matrix->JA[idx]];
            }
            output_vector[row] = sum;
        }
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

void matvec_parallel_csr(CSR_matrix *csr_matrix, double *x, double *y, struct performance *node, int *thread_numbers, struct performance **head, struct performance **tail, int new_non_zero_values, double *effective_results)
{
    double time_used, time = 0.0;
    for (int index = 0; index < NUM_THREADS; index++)
    {
        int num_threads = thread_numbers[index];
        int actual_threads;

        int *first_row = (int *)calloc(num_threads + 1, sizeof(int));
        if (!first_row)
        {
            printf("Errore in calloc per first_row\n");
            exit(EXIT_FAILURE);
        }

        first_row = matrix_partition(csr_matrix, num_threads, &actual_threads);

        for (int k = 0; k < COMPUTATION; k++)
        {
            product(csr_matrix, x, y, actual_threads, &time_used, first_row);
            if (!compute_norm(y, effective_results, csr_matrix->M, 1e-4))
            {
                printf("Error in check for %s after parallel csr\n", csr_matrix->name);
                sleep(3);
            }
            time += time_used;
            re_initialize_y_vector(csr_matrix->M, y);
        }

        time = time / COMPUTATION;
        compute_parallel_performance(node, &time, new_non_zero_values, num_threads);
        printf("Time used for execution for %d with csr with OpenMP = %.16lf\n", num_threads, node->time_used);

        node->non_zeroes_values = new_non_zero_values;
        node->computation = PARALLEL_OPEN_MP_CSR;

        add_node_performance(head, tail, node);

        print_parallel_csr_result(node);
        node = NULL;
        node = reset_node();
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

    printf("Time used for hll with OpenMP with %d thread = %.16lf\n", num_threads, node->time_used);
}

// Matrix-vector parallel dot product in HLL format
void matvec_parallel_hll(HLL_matrix *hll_matrix, double *x, double *y, struct performance *node, int *thread_numbers, struct performance **head, struct performance **tail, int new_non_zero_values, double *effective_results)
{
    double time_used, time = 0.0, start, end;
    for (int index = 0; index < NUM_THREADS; index++)
    {
        int num_threads = thread_numbers[index];

        double *times_vector = (double *)calloc(num_threads, sizeof(double));
        if (!times_vector)
        {
            perror("Error in calloc per times_vector");
            exit(EXIT_FAILURE);
        }

        for (int k = 0; k < COMPUTATION; k++)
        {
            hll_parallel_product(hll_matrix, x, y, num_threads, &time_used);
            if (!compute_norm(y, effective_results, hll_matrix->M, 1e-4))
            {
                printf("Error in check for %s after hll with OpenMP\n", hll_matrix->name);
                sleep(3);
            }
            re_initialize_y_vector(hll_matrix->M, y);
            time += time_used;
        }

        time = time / COMPUTATION;
        compute_hll_parallel_performance(node, new_non_zero_values, time, num_threads);
        print_parallel_hll_result(node);

        add_node_performance(head, tail, node);

        re_initialize_y_vector(hll_matrix->M, y);

        node = NULL;
        node = reset_node();
        strcpy(node->matrix, hll_matrix->name);
    }
}