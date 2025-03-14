#ifndef HLL_HEADERS_H
#define HLL_HEADERS_H

#include "../data_structures/hll_matrix.h"
#include "../data_structures/ellpack_block.h"
#include "../data_structures/csr_matrix.h"
#include "../headers/matrix_format.h"

void read_HLL_matrix(matrix_format *matrix, HLL_matrix *hll_matrix);
void print_HLL_matrix(HLL_matrix *matrix);
void destroy_HLL_matrix(HLL_matrix *matrix);

#endif