#ifndef ELLPACK_HEADERS_H
#define ELLPACk_HEADERS_H

#include "../data_structures/ellpack_matrix.h"

ELLPACK_matrix *read_ELLPACK_matrix(char *filename);
void print_ELLPACK_matrix(ELLPACK_matrix *matrix);
void destroy_ELLPACK_matrix(ELLPACK_matrix *matrix);

#endif