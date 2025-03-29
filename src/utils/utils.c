#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "../data_structures/csr_matrix.h"
#include "../data_structures/hll_matrix.h"
#include "../data_structures/performance.h"
#include "../headers/matrix_format.h"
#include "../utils_header/computation_type.h"
#include "../utils_header/utils.h"
#include "../utils_header/mmio.h"

#define BASE_DIR "data/"
#define IS_ZERO(x) (fabs(x) == 0.0)

struct vec3d
{
    int row;
    int col;
    int pos;
};

static int readline(FILE *f, struct vec3d *item, double *val, matrix_format *matrix)
{
    if (!mm_is_pattern(matrix->matrix_typecode))
    {
        fscanf(f, "%d %d %lg", &item->row, &item->col, val);
    }
    else
    {
        fscanf(f, "%d %d", &item->row, &item->col);
        *val = 1.0;
    }
    item->row -= 1;
    item->col -= 1;
    return !IS_ZERO(*val);
}

static inline int lt(struct vec3d *a, struct vec3d *b)
{
    if ((a->row < b->row) || (a->row == b->row) && (a->col < b->col))
    {
        return -1;
    }
    else if ((a->row == b->row) && (a->col == b->col))
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

static inline int qcmp(const void *a, const void *b)
{
    return lt((struct vec3d *)a, (struct vec3d *)b);
}

static int __parse_rows(FILE *f, matrix_format *matrix)
{
    int err;
    int is_symmetric = mm_is_symmetric(matrix->matrix_typecode);
    struct vec3d *items = malloc((is_symmetric ? 2 : 1) * matrix->number_of_non_zeoroes_values * sizeof(struct vec3d));
    if (!items)
    {
        err = -1;
        goto no_items;
    }
    double *packed_data = malloc(matrix->number_of_non_zeoroes_values * sizeof(double));
    if (!packed_data)
    {
        err = -1;
        goto no_pck_data;
    }
    int real_nz = 0;
    int explicit_zeros = 0;
    for (int k = 0; k < matrix->number_of_non_zeoroes_values; k++)
    {
        if (readline(f, &items[real_nz], &packed_data[k], matrix))
        {
            int i = items[real_nz].row;
            int j = items[real_nz].col;
            items[real_nz].pos = k;
            ++real_nz;
            if (is_symmetric && i != j)
            {
                items[real_nz].row = j;
                items[real_nz].col = i;
                items[real_nz].pos = k;
                ++real_nz;
            }
        }
        else
        {
            ++explicit_zeros;
        }
    }

    qsort(items, real_nz, sizeof(struct vec3d), qcmp);
    int *rows = malloc(real_nz * sizeof(int));
    if (!rows)
    {
        err = -1;
        goto no_rows;
    }
    int *cols = malloc(real_nz * sizeof(int));
    if (!cols)
    {
        err = -1;
        goto no_cols;
    }
    double *data;
    if (mm_is_symmetric(matrix->matrix_typecode))
    {
        data = malloc(real_nz * sizeof(double));
        if (!data)
        {
            err = -1;
            goto no_data;
        }
        for (int i = 0; i < real_nz; ++i)
        {
            data[i] = packed_data[items[i].pos];
        }
    }
    else
    {
        data = packed_data;
    }
    for (int i = 0; i < real_nz; ++i)
    {
        rows[i] = items[i].row;
        cols[i] = items[i].col;
    }
    matrix->rows = rows;
    matrix->columns = cols;
    matrix->values = data;
    matrix->number_of_non_zeoroes_values = real_nz;
    return 0;
no_data:
    free(cols);
no_cols:
    free(rows);
no_rows:
    free(packed_data);
no_pck_data:
    free(items);
no_items:
    return err;
}

static int parse_rows_sy(FILE *f, matrix_format *matrix)
{
    return __parse_rows(f, matrix);
}

static int parse_rows_ns(FILE *f, matrix_format *matrix)
{
    return __parse_rows(f, matrix);
}

static int parse_rows(FILE *f, matrix_format *matrix)
{
    if (mm_is_symmetric(matrix->matrix_typecode))
    {
        return parse_rows_sy(f, matrix);
    }
    else
    {
        return parse_rows_ns(f, matrix);
    }
}

// Funzione per leggere le matrici
int read_matrix(FILE *matrix_file, matrix_format *matrix)
{
    if (mm_read_banner(matrix_file, &(matrix->matrix_typecode)) != 0)
    {
        printf("could not process Matrix Market banner\n");
        return 1;
    }

    if (mm_is_complex(matrix->matrix_typecode) && mm_is_matrix(matrix->matrix_typecode) && mm_is_sparse(matrix->matrix_typecode))
    {
        printf("sorry, this application does not support ");
        printf("matrix market type: [%s]\n", mm_typecode_to_str(matrix->matrix_typecode));
        return 1;
    }

    int M, N, nz;
    int ret_code = mm_read_mtx_crd_size(matrix_file, &M, &N, &nz);
    if (ret_code)
    {
        return 1;
    }

    matrix->M = M;
    matrix->N = N;
    matrix->number_of_non_zeoroes_values = nz;
    int ir = parse_rows(matrix_file, matrix);
    fclose(matrix_file);
    return ir;
}

FILE *get_matrix_file(const char *dir_name, char *matrix_filename)
{
    // Varibles to read the matrix market file
    MM_typecode matcode;
    char matrix_fullpath[256];
    int is_sparse_matrix, is_array_file;

    // Get the full path of the matrix market file to open
    snprintf(matrix_fullpath, sizeof(matrix_fullpath), "%s/%s", dir_name, matrix_filename);

    // Open the file
    FILE *matrix_file = fopen(matrix_fullpath, "r");
    if (matrix_file == NULL)
    {
        printf("Error occurred while opening matrix file: %s\nError code: %d\n", matrix_filename, errno);
        exit(EXIT_FAILURE);
    }

    return matrix_file;
}

void mtx_cleanup(matrix_format *matrix)
{
    free(matrix->columns);
    free(matrix->rows);
    free(matrix->values);
}

double compute_norm(double *z, double *y, int n, double esp)
{
    double s = 0.0;

    for (int i = 0; i < n; i++)
    {
        double d = fabs(z[i] - y[i]);
        if (d >= esp)
        {
            s += d;
        }
    }
    s /= n;
    if (s > 0.0)
        printf("error = %.16lf\n", s);
    return s > 0.0 ? 0 : 1;
}

double *unit_vector(int size)
{
    double *x = (double *)malloc(size * sizeof(double));
    if (!x)
    {
        printf("Error in malloc per x\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++)
        x[i] = 1.0;

    return x;
}

void add_node_performance(struct performance *head, struct performance *tail, struct performance *node)
{
    if (head == NULL)
    {
        head = (struct performance *)calloc(1, sizeof(struct performance));
        if (head == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
        }
        tail = (struct performance *)calloc(1, sizeof(struct performance));
        if (tail == NULL)
        {
            printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
        }

        head = node;
        tail = node;
    }
    else
    {
        tail->next_node = node;
        node->prev_node = tail;
        tail = node;
    }
}

void *reset_node(struct performance *node)
{
    node = NULL;
    node = (struct performance *)calloc(1, sizeof(struct performance));
    if (node == NULL)
    {
        printf("Error occour in calloc for performance node\nError Code: %d\n", errno);
        exit(EXIT_FAILURE);
    }
}

void compute_serial_performance(struct performance *node, double time_used, int new_non_zero_values)
{
    // Compute metrics
    double flops = 2.0 * new_non_zero_values / time_used;
    double mflops = flops / 1e6;
    double gflops = flops / 1e9;

    node->number_of_threads_used = 1;
    node->flops = flops;
    node->mflops = mflops;
    node->gflops = gflops;
    node->time_used = time_used;
}

void print_serial_csr_result(struct performance *node)
{
    printf("Prestazioni Ottenute con il prodotto utilizzando il formato csr!\n");
    printf("\n\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
    printf("Time used for dot-product:      %.16lf\n", node->time_used);
    printf("FLOPS:                          %.16lf\n", node->flops);
    printf("MFLOPS:                         %.16lf\n", node->mflops);
    printf("GFLOPS:                         %.16lf\n\n", node->gflops);
}

void print_serial_hll_result(struct performance *node)
{
    printf("Prestazioni Ottenute con il prodotto utilizzando il formato hll in modalità seriale!\n");
    printf("\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
    printf("Time used for dot-product:      %.16lf\n", node->time_used);
    printf("FLOPS:                          %.16lf\n", node->flops);
    printf("MFLOPS:                         %.16lf\n", node->mflops);
    printf("GFLOPS:                         %.16lf\n\n", node->gflops);
}

void print_parallel_csr_result(struct performance *node)
{
    printf("\n\nParallel Performance for %s with %d threads, with CSR:\n", node->matrix, node->number_of_threads_used);
    printf("Time used for dot-product:      %.16lf\n", node->time_used);
    printf("FLOPS:                          %.16lf\n", node->flops);
    printf("MFLOPS:                         %.16lf\n", node->mflops);
    printf("GFLOPS:                         %.16lf\n\n", node->gflops);
}

void print_parallel_hll_result(struct performance *node)
{
    printf("\n\nParallel Performance for %s with %d threads, with HLL:\n", node->matrix, node->number_of_threads_used);
    printf("Time used for dot-product:      %.16lf\n", node->time_used);
    printf("FLOPS:                          %.16lf\n", node->flops);
    printf("MFLOPS:                         %.16lf\n", node->mflops);
    printf("GFLOPS:                         %.16lf\n\n", node->gflops);
}

void print_cuda_hll_kernel_performance(struct performance *node)
{
    printf("\n\nPrestazioni Ottenute con il prodotto utilizzando il formato hll con CUDA!\n");
    switch (node->computation)
    {
    case CUDA_HLL_KERNEL_1:
        printf("CUDA HLL Kernel 1\n");
        break;

    case CUDA_HLL_KERNEL_2:
        printf("CUDA HLL Kernel 2\n");
        break;

    case CUDA_HLL_KERNEL_3:
        printf("CUDA HLL Kernel 3\n");
        break;

    case CUDA_HLL_KERNEL_4:
        printf("CUDA HLL Kernel 4\n");
        break;
    }

    printf("\nPerformance for %s with %d threads:\n", node->matrix, node->number_of_threads_used);
    printf("Time used for dot-product:      %.16lf\n", node->time_used);
    printf("FLOPS:                          %.16lf\n", node->flops);
    printf("MFLOPS:                         %.16lf\n", node->mflops);
    printf("GFLOPS:                         %.16lf\n\n", node->gflops);
}

void print_cuda_csr_kernel_performance(struct performance *node)
{
    printf("\n\nPrestazioni Ottenute con il prodotto utilizzando il formato CSR con CUDA!\n");
    switch (node->computation)
    {
    case CUDA_CSR_KERNEL_1:
        printf("CUDA CSR Kernel 1\n");
        break;

    case CUDA_CSR_KERNEL_2:
        printf("CUDA CSR Kernel 2\n");
        break;

    case CUDA_CSR_KERNEL_3:
        printf("CUDA CSR Kernel 3\n");
        break;

    case CUDA_CSR_KERNEL_4:
        printf("CUDA CSR Kernel 4\n");
        break;
    }

    printf("\nPerformance for %s with %d threads:", node->matrix, node->number_of_threads_used);
    printf("Time used for dot-product:      %.16lf\n", node->time_used);
    printf("FLOPS:                          %.16lf\n", node->flops);
    printf("MFLOPS:                         %.16lf\n", node->mflops);
    printf("GFLOPS:                         %.16lf\n\n", node->gflops);
}

void print_list(struct performance *head)
{
    while (head != NULL)
    {

        printf("Numero di thread:           %d\n", head->number_of_threads_used);
        switch (head->computation)
        {
        case SERIAL_CSR:
            printf("Serial CSR\n");
            break;

        case SERIAL_HHL:
            printf("Serial HLL\n");
            break;

        case PARALLEL_OPEN_MP_CSR:
            printf("Parallel OPEN MP CSR\n");
            break;

        case PARALLEL_OPEN_MP_HLL:
            printf("Parallel OPEN MP CSR\n");
            break;

        case CUDA_CSR_KERNEL_1:
            printf("CUDA CSR KERNEL 1\n");
            break;

        case CUDA_CSR_KERNEL_2:
            printf("CUDA CSR KERNEL 2\n");
            break;

        case CUDA_CSR_KERNEL_3:
            printf("CUDA CSR KERNEL 3\n");
            break;

        case CUDA_CSR_KERNEL_4:
            printf("CUDA CSR KERNEL 4\n");
            break;

        case CUDA_HLL_KERNEL_1:
            printf("CUDA HLL KERNEL 1\n");
            break;

        case CUDA_HLL_KERNEL_2:
            printf("CUDA HLL KERNEL 2\n");
            break;

        case CUDA_HLL_KERNEL_3:
            printf("CUDA HLL KERNEL 3\n");
            break;

        case CUDA_HLL_KERNEL_4:
            printf("CUDA HLL KERNEL 4\n");
            break;
        }
        printf("Time Used:                  %.16lf\n", head->time_used);
        printf("Flops:                      %.16lf\n", head->flops);
        printf("MFLOPS:                     %.16lf\n", head->mflops);
        printf("GFLOPS:                     %.16lf\n", head->gflops);
        head = head->next_node;
    }
}