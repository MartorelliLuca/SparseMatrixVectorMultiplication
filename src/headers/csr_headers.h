#ifndef CSR_HEADERS_H
#define CSR_HEADERS_H

#include "../data_structures/csr_matix.h"

csr_matrix *read_csr_matrix(char *filename);
void print_csr_matrix(csr_matrix *matrix);
void destroy_csr_matrix(csr_matrix *matrix);

#endif