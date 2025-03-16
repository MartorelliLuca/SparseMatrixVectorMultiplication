#ifndef UTILS_H
#define UTILS_H

#include "../data_structures/csr_matrix.h"
#include "../data_structures/hll_matrix.h"
#include "../headers/matrix_format.h"

FILE *get_matrix_file(const char *dir_name, char *matrix_filename);
int read_matrix(FILE *matrix_file, matrix_format *matrix);
void mtx_cleanup(matrix_format *matrix);
double compute_norm(double *z, double *y, int n, double esp);

#endif