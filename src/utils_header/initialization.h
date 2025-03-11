#ifndef INIZIALITAZION_H
#define INIZIALITAZION_H

#include "../data_structures/csr_matrix.h"
#include "../data_structures/hll_matrix.h"
#include "../headers/matrix.h"

FILE *get_matrix_file(char *dir_name, char *matrix_filename);
double *initialize_x_vector(int size);
double *initialize_y_vector(int size);
void re_initialize_y_vector(int size, double *y);
double **get_matrix_from_csr(CSR_matrix *csr_matrix);
void free_dense_matrix(double **dense_matrix, int M);
void get_matrix_format(FILE *matrix_file, int *file_type, matrix_format *matrix);

#endif