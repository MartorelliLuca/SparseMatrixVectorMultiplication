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

void matrix_partition(CSR_matrix *CSR_matrix, int nz, int num_threads, int *first_row)
{
    int *row_nnz = (int *)malloc(CSR_matrix->M * sizeof(int));
    int *nnz_per_thread_count = (int *)calloc(num_threads, sizeof(int));
    if (row_nnz == NULL || nnz_per_thread_count == NULL)
    {
        printf("Error occour in malloc for row_nnz or nnz_per_thread_count\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }

    double total_nnz = 0.0;
    for (int i = 0; i < CSR_matrix->M; i++)
    {
        row_nnz[i] = CSR_matrix->IRP[i + 1] - CSR_matrix->IRP[i];
        total_nnz += row_nnz[i];
    }

    double target_workload = total_nnz / num_threads;
    double current_workload = 0.0;
    int thread_id = 0;
    first_row[0] = 0;

    // Suddivisione migliorata
    for (int i = 0; i < CSR_matrix->M; i++)
    {
        current_workload += row_nnz[i];
        nnz_per_thread_count[thread_id] += row_nnz[i];

        // Se il thread ha accumulato abbastanza lavoro e ci sono ancora thread disponibili
        if (current_workload >= target_workload && thread_id < num_threads - 1)
        {
            first_row[thread_id + 1] = i + 1; // Il prossimo thread partirà da qui
            thread_id++;
            current_workload = 0.0;
        }
    }

    first_row[num_threads] = CSR_matrix->M;

    free(row_nnz);
    free(nnz_per_thread_count);
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

void product(CSR_matrix *csr_matrix, double *x, double *y, int num_threads, int *first_row, double *time_used)
{
    int my_thread_number = omp_get_thread_num();
    int row_one = first_row[my_thread_number];
    int final_row = (my_thread_number == num_threads - 1) ? csr_matrix->M : first_row[my_thread_number + 1];
    double start = omp_get_wtime();

#pragma omp parallel for

    for (int i = row_one; i < final_row; i++)
    {
        for (int j = csr_matrix->IRP[i]; j < csr_matrix->IRP[i + 1]; j++)
        {
            y[i] += csr_matrix->AS[j] * x[csr_matrix->JA[j]];
        }
    }

    double end = omp_get_wtime();
    *time_used = end - start;
    printf("Thread %d: Tempo impiegato %.16lf\n", my_thread_number, *time_used);
}

// TODO cambia il modo in cui calcoli performance
void matvec_parallel_csr(CSR_matrix *csr_matrix, double *x, double *y, struct performance *node, int *thread_numbers, struct performance *head, struct performance *tail, int new_non_zero_values)
{

    double time_used, start, end;
    for (int index = 0; index < 6; index++)
    {
        int num_threads = thread_numbers[index];

        // Set number of threads to perform dot product executions
        omp_set_num_threads(num_threads);

        // partiziona il lavoro tra i thread
        int *first_row;

        first_row = (int *)calloc(num_threads + 1, sizeof(int)); // Allocare num_threads + 1 elementi
        if (!first_row)
        {
            perror("Errore in calloc per first_row");
            exit(EXIT_FAILURE);
        }

        // TODO GUARDA QUESTE

        matrix_partition(csr_matrix, new_non_zero_values, num_threads, first_row);
        product(csr_matrix, x, y, num_threads, first_row, &time_used);

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
        int start_row = h * HACKSIZE;
        int end_row = (start_row + HACKSIZE < hll_matrix->M) ? start_row + HACKSIZE : hll_matrix->M;
        int maxNR = (hll_matrix->hack_offsets[h + 1] - hll_matrix->hack_offsets[h]) / (end_row - start_row);

        int block_offset = hll_matrix->hack_offsets[h];

        for (int i = start_row; i < end_row; i++)
        {
            double sum = 0.0;
            int row_offset = block_offset + (i - start_row) * maxNR;

            for (int j = 0; j < maxNR; j++)
            {
                int col = hll_matrix->JA[row_offset + j];
                double val = hll_matrix->AS[row_offset + j];

                if (col >= 0)
                { // Verifica che la colonna sia valida
                    sum += val * x[col];
                }
            }
            y[i] = sum;
        }
    }
}

int get_real_non_zero_values_count(CSR_matrix *matrix)
{
    int total_elements = matrix->M * matrix->N;
    int non_zero_elements = matrix->NZ;
    printf("Elementi totali: %d\n", total_elements);
    printf("Elementi NON nulli: %d\n", non_zero_elements);
    printf("Elementi nulliAAAAAAAA: %d\n", total_elements - non_zero_elements);
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
}
