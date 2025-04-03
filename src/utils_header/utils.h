#ifndef UTILS_H
#define UTILS_H

#include "../data_structures/csr_matrix.h"
#include "../data_structures/hll_matrix.h"
#include "../data_structures/performance.h"
#include "../headers/matrix_format.h"
#include "../utils_header/computation_type.h"

FILE *get_matrix_file(const char *dir_name, char *matrix_filename);

int read_matrix(FILE *matrix_file, matrix_format *matrix);
void mtx_cleanup(matrix_format *matrix);
double compute_norm(double *z, double *y, int n, double esp);
double *unit_vector(int size);

void add_node_performance(struct performance **head, struct performance **tail, struct performance *node);
void print_serial_csr_result(struct performance *node);
void print_serial_hll_result(struct performance *node);

void print_parallel_csr_result(struct performance *node);
void print_parallel_hll_result(struct performance *node);

void print_cuda_hll_kernel_performance(struct performance *node);
void print_cuda_csr_kernel_performance(struct performance *node);

void print_list(struct performance *head);

struct performance *reset_node();
void compute_serial_performance(struct performance *node, double time_used, int new_non_zero_values);

void save_performance_to_csv(struct performance *head);
void free_performance_list(struct performance **head);

void print_progress_bar(char *s, double part, double total, double oldPart);
void print_title();

#endif