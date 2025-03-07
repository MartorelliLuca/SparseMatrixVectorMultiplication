#ifndef ELLPACK_HEADERS_H
#define ELLPACk_HEADERS_H

#include "../data_structures/ellpack_matrix.h"

ellpack_matrix *read_ellpack_matrix(char *filename);
void print_ellpack_matrix(ellpack_matrix *matrix);
void destroy_ellpack_matrix(ellpack_matrix *matrix);

#endif