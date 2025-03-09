#ifndef HLL_HEADERS_H
#define HLL_HEADERS_H

#include "../data_structures/hll_matrix.h"

void *read_HLL_matrix(FILE *matrix_file, HLL_matrix *hll_matrix, int *file_type);
void print_HLL_matrix(HLL_matrix *matrix);
void destroy_HLL_matrix(HLL_matrix *matrix);

#endif