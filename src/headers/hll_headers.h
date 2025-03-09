#ifndef HLL_HEADERS_H
#define HLL_HEADERS_H

#include "../data_structures/hll_matrix.h"

HLL_matrix *read_ELLPACK_matrix(char *filename);
void convert_to_ellpack(HLL_matrix *ellpack_matrix);
void print_ELLPACK_matrix(HLL_matrix *matrix);
void destroy_ELLPACK_matrix(HLL_matrix *matrix);

#endif