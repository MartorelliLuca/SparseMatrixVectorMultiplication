#ifndef CSR_HEADERS_H
#define CSR_HEADERS_H

#include "../data_structures/csr_matix.h"

void read_CSR_matrix(FILE *matrix_file, CSR_matrix *csr_matrix);
void convert_to_csr(CSR_matrix *CSR_matrix);
void print_CSR_matrix(CSR_matrix *csr_matrix);
void destroy_CSR_matrix(CSR_matrix *csr_matrix);

#endif