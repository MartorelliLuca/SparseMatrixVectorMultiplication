#ifndef OPERATION_H
#define OPERATION_H

#include "../data_structures/csr_matrix.h"
#include "../data_structures/hll_matrix.h"

void matvec(double **A, double *x, double *y, int M, int N);
void matvec_serial_csr(CSR_matrix *csr_matrix, double *x, double *y);

void matvec_parallel_csr(CSR_matrix *csr_matrix, double *x, double *y, struct performance *node, int *thread_numbers,
                         struct performance *head, struct performance *tail, int new_non_zero_values);

void matvec_serial_hll(HLL_matrix *ellpack_matrix, double *x, double *y);
void compute_serial_performance(struct performance *node, double time_used, int new_non_zero_values);
void compute_parallel_performance(struct performance *node, double time_used, int new_non_zero_values, int thread_numbers);

#endif