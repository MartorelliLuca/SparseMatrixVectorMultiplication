#ifndef CSR_HEADERS_H
#define CSR_HEADERS_H

#include "../data_structures/csr_matrix.h"
#include "matrix_format.h"

void read_CSR_matrix(FILE *matrix_file, CSR_matrix *csr_matrix, matrix_format *matrix);
void destroy_CSR_matrix(CSR_matrix *csr_matrix);

#endif