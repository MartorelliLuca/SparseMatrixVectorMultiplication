#ifndef CSR_HEADERS_H
#define CSR_HEADERS_H

#include "../data_structures/csr_matix.h"

CSR_matrix *read_CSR_matrix(char *filename);
void destroy_CSR_matrix(CSR_matrix *csr_matrix);

#endif