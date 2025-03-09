#ifndef INIZIALITAZION_H
#define INIZIALITAZION_H

#include "../data_structures/csr_matix.h"
#include "../data_structures/hll_matrix.h"

FILE *get_matrix_file(char *dir_name, char *matrix_filename, int *file_type, int *symmetric);
double *initialize_x_vector(int size);
double *initialize_y_vector(int size);
double **get_matrix_from_csr(CSR_matrix *csr_matrix);
void free_dense_matrix(double **dense_matrix, int M);

#endif