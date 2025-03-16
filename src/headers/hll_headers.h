#ifndef HLL_HEADERS_H
#define HLL_HEADERS_H

#include "../data_structures/hll_matrix.h"
#include "../data_structures/csr_matrix.h"
#include "../headers/matrix_format.h"

void read_HLL_matrix(HLL_matrix *hll, int hack_size, matrix_format *matrix);
void print_HLL_matrix(const HLL_matrix *matrix);
void destroy_HLL_matrix(HLL_matrix *matrix);

#endif