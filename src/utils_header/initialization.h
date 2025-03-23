#ifndef INIZIALITAZION_H
#define INIZIALITAZION_H

#include "../data_structures/csr_matrix.h"
#include "../data_structures/hll_matrix.h"
#include "../headers/matrix_format.h"

double *initialize_x_vector(int size);
double *initialize_y_vector(int size);
void re_initialize_y_vector(int size, double *y);

double **get_matrix_from_csr(CSR_matrix *csr_matrix);
void free_dense_matrix(double **dense_matrix, int M);
void get_matrix_format(FILE *matrix_file, int *file_type, matrix_format *matrix);

int *initialize_threads_number();
CSR_matrix *get_csr_matrix();
HLL_matrix *get_hll_matrix();
matrix_format *get_matrix_format_matrix();

#endif