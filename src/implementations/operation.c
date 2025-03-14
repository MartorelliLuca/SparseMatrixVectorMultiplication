#include <ctype.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <bits/time.h>
#include <errno.h>

#include "../data_structures/csr_matrix.h"
#include "../data_structures/hll_matrix.h"
#include "../data_structures/performance.h"

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

void product(CSR_matrix *csr_matrix, double *input_vector, double *output_vector, int num_threads, int *start_row_indices, double *execution_time, double *times_vector)
{

#pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int end_row = (thread_id == num_threads - 1) ? csr_matrix->M : start_row_indices[thread_id + 1];

        double local_start_time = omp_get_wtime();

        for (int row = start_row_indices[thread_id]; row < end_row; row++)
        {
            double sum = 0.0;
            for (int idx = csr_matrix->IRP[row]; idx < csr_matrix->IRP[row + 1]; idx++)
            {
                sum += csr_matrix->AS[idx] * input_vector[csr_matrix->JA[idx]];
            }
            output_vector[row] = sum;
        }

        double local_end_time = omp_get_wtime();
        printf("Thread %d: started: %.16lf, ended: %.16lf and took this time: %.16lf\n", thread_id, local_start_time, local_end_time, local_end_time - local_start_time);
        *execution_time = local_end_time - local_start_time;
        times_vector[thread_id] = local_end_time - local_start_time;
    }
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

// void print_vector(double *y, int size)
// {
//     for (int i = 0; i < size; i++)
//     {
//         printf("%.6lf ", y[i]);
//     }
//     printf("\n");
// }

void matvec_parallel_csr(CSR_matrix *csr_matrix, double *x, double *y, struct performance *node, int *thread_numbers, struct performance *head, struct performance *tail, int new_non_zero_values)
{

    double time_used, start, end;
    for (int index = 0; index < 6; index++)
    {
        int num_threads = thread_numbers[index];

        // Set number of threads to perform dot product executions

        // partiziona il lavoro tra i thread
        int *first_row;

        first_row = (int *)calloc(num_threads + 1, sizeof(int)); // Allocare num_threads + 1 elementi
        if (!first_row)
        {
            perror("Errore in calloc per first_row");
            exit(EXIT_FAILURE);
        }

        // TODO GUARDA QUESTE

        double *times_vector = (int *)calloc(num_threads, sizeof(double));
        if (!times_vector)
        {
            perror("Errore in calloc per times_vector");
            exit(EXIT_FAILURE);
        }

        matrix_partition(csr_matrix, num_threads, first_row);
        product(csr_matrix, x, y, num_threads, first_row, &time_used, times_vector);

        compute_parallel_performance(node, time_used, times_vector, new_non_zero_values, num_threads);

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

        // fai il prodotto scalare tra la riga i-esima e il vettore x
    }
}

// First attempt to do matrix-vector dot produt in HLL format
void matvec_serial_hll(HLL_matrix *hll_matrix, double *x, double *y)
{
    int index = 0;
    for (int b = 0; b < hll_matrix->num_hack; b++)
    {
        int max_nz = hll_matrix->max_non_zeroes[b];
        for (int r = 0; r < hll_matrix->hack_size && (b * hll_matrix->hack_size + r) < hll_matrix->M; r++)
        {
            for (int c = 0; c < max_nz; c++)
            {
                int col_idx = hll_matrix->columns[index];
                double val = hll_matrix->values[index];
                if (col_idx != -1)
                {
                    y[b * hll_matrix->hack_size + r] += val * x[col_idx];
                }
                index++;
            }
        }
    }
}

int get_real_non_zero_values_count(CSR_matrix *matrix)
{
    int total_elements = matrix->M * matrix->N;
    int non_zero_elements = matrix->NZ;
    // printf("Elementi totali: %d\n", total_elements);
    // printf("Elementi NON nulli: %d\n", non_zero_elements);
    // printf("Elementi nulliAAAAAAAA: %d\n", total_elements - non_zero_elements);
    return total_elements - non_zero_elements;
}

void compute_serial_performance(struct performance *node, double time_used, int new_non_zero_values)
{
    // Compute metrics
    double flops = 2.0 * new_non_zero_values / time_used;
    double mflops = flops / 1e6;
    double gflops = flops / 1e9;

    node->number_of_threads_used = 1;
    node->flops = flops;
    node->mflops = mflops;
    node->gflops = gflops;
    node->time_used = time_used;
}

void compute_parallel_performance(struct performance *node, double time_, double *times_vector, int new_non_zero_values, int num_threads)
{
    double max_time = 0.0;
    for (int i = 0; i < num_threads; i++)
    {
        if (times_vector[i] > max_time)
        {
            max_time = times_vector[i];
        }
    }
    printf("Max time: %.16lf\n", max_time);
    printf("Tempo totale impiegato: %.16lf\n", max_time);
    // Compute metrics

    double flops = 2.0 * new_non_zero_values / max_time;
    double mflops = flops / 1e6;
    double gflops = flops / 1e9;

    node->number_of_threads_used = num_threads;
    node->flops = flops;
    node->mflops = mflops;
    node->gflops = gflops;
    node->time_used = max_time;
}
