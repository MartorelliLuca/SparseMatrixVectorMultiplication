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

// void matrix_partition(int M, int nz, int *num_threads, int *IRP, int *first_row, int *last_row)
// {
//     int block_size = M / nz;
//     for (int i = 0; i < nz; i++)
//     {
//         first_row[i] = i * block_size;
//         last_row[i] = (i + 1) * block_size;
//     }
//     last_row[nz - 1] = M;
// }

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

void product(CSR_matrix *csr_matrix, double *x, double *y, int num_threads, int *first_row, int *last_row)
{
#pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        for (int i = first_row[tid]; i < last_row[tid]; i++)
        {
            for (int j = csr_matrix->IRP[i]; j < csr_matrix->IRP[i + 1]; j++)
            {
                y[i] += csr_matrix->AS[j] * x[csr_matrix->JA[j]];
            }
        }
    }
}

// TODO cambia il modo in cui calcoli performance
void matvec_parallel_csr(CSR_matrix *csr_matrix, double *x, double *y, struct performance *node,
                         int *thread_numbers, struct performance *head, struct performance *tail)
{

    double time_used, start, end;
    int new_non_zero_values;
    for (int index = 0; index < 6; index++)
    {
        int num_threads = thread_numbers[index];

        // Set number of threads to perform dot product executions
        omp_set_num_threads(num_threads);

        // partiziona il lavoro tra i thread
        int *first_row, *last_row;

        // matrix_partition(csr_matrix->M, num_threads, first_row, last_row);

        product(csr_matrix, x, y, num_threads, first_row, last_row);

        compute_parallel_performance(node, time_used, new_non_zero_values, num_threads);

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
    for (int h = 0; h < hll_matrix->num_hacks; h++)
    {
        int start_idx = hll_matrix->hack_offsets[h];
        int end_idx = (h < hll_matrix->num_hacks - 1) ? hll_matrix->hack_offsets[h + 1] : hll_matrix->M;
        int hack_rows = end_idx - start_idx;

        for (int i = 0; i < hack_rows; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < hll_matrix->hack_size; j++)
            {
                int index = start_idx * hll_matrix->hack_size + i * hll_matrix->hack_size + j;
                int col = hll_matrix->JA[index];
                double val = hll_matrix->AS[index];

                if (col >= 0)
                {
                    sum += val * x[col];
                }
            }
            y[start_idx + i] += sum;
        }
    }
}

int get_real_non_zero_values_count(CSR_matrix *matrix)
{
    int count = 0;

    for (int i = 0; i < matrix->M; i++)
    {
        for (int j = matrix->IRP[i]; j < matrix->IRP[i + 1]; j++)
        {
            count++;

            // If the matrix is symmetrical, we consider the mirror value only once
            if (matrix->is_symmetric && matrix->JA[j] > i)
            {
                count++;
            }
        }
    }

    return count;
}

void compute_serial_performance_csr(struct performance *node, double time_used, int new_non_zero_values)
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

void compute_parallel_performance(struct performance *node, double time_used, int new_non_zero_values, int num_threads)
{
    // Compute metrics

    double flops = 2.0 * new_non_zero_values / time_used;
    double mflops = flops / 1e6;
    double gflops = flops / 1e9;

    node->number_of_threads_used = num_threads;
    node->flops = flops;
    node->mflops = mflops;
    node->gflops = gflops;
    node->time_used = time_used;
    printf("///////////////////////////\n///////////////////////////\n");
    printf("Time used in node:      %.16lf\n", node->time_used);
    printf("FLOPS:                          %.16lf\n", node->flops);
    printf("MFLOPS:                         %.16lf\n", node->mflops);
    printf("GFLOPS:                         %.16lf\n\n", node->gflops);
    printf("new_non_zero_values: %d\n", new_non_zero_values);
    printf("time_used: %lf\n", time_used);
}
