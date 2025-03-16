#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <string.h>

#include "../data_structures/hll_matrix.h"
#include "../utils_header/initialization.h"
#include "../data_structures/csr_matrix.h"
#include "../headers/hll_headers.h"
#include "../headers/csr_headers.h"
#include "../headers/matrix_format.h"

static inline int max(int *v, int lo, int hi_exclusive)
{
    int max = -1;
    for (int i = lo; i < hi_exclusive; ++i)
    {
        if (max < v[i])
        {
            max = v[i];
        }
    }
    return max;
}

static void get_nonzeros(matrix_format *matrix, int *nzr)
{
    memset(nzr, 0, sizeof(int) * matrix->M);
    for (int i = 0; i < matrix->number_of_non_zeoroes_values; ++i)
    {
        nzr[matrix->rows[i]]++;
    }
}

static int get_data_size(int *nzr, int hack_size, int num_rows)
{
    int data_size = 0;
    int start = 0;
    while (start + hack_size < num_rows)
    {
        data_size += hack_size * max(nzr, start, start + hack_size);
        start += hack_size;
    }
    return data_size + (num_rows - start) * max(nzr, start, num_rows);
}

static inline void ifill(int *v, int start, int n, int val)
{
    for (int i = 0; i < n; ++i)
    {
        v[start + i] = val;
    }
}

static inline void dfill(double *v, int start, int n, double val)
{
    for (int i = 0; i < n; ++i)
    {
        v[start + i] = val;
    }
}

static void read_row(matrix_format *matrix, int *mmp, HLL_matrix *hll, int *hllp, int maxnzr)
{
    int row = matrix->rows[*mmp];
    int n = 0;
    for (int i = *mmp; i < matrix->number_of_non_zeoroes_values && row == matrix->rows[i]; ++i)
    {
        ++n;
    }

    memcpy(&hll->data[*hllp], &matrix->values[*mmp], n * sizeof(double));
    memcpy(&hll->col_index[*hllp], &matrix->columns[*mmp], n * sizeof(int));
    *hllp += n;
    *mmp += n;

    ifill(hll->col_index, *hllp, maxnzr - n, hll->col_index[*hllp - 1]);
    dfill(hll->data, *hllp, maxnzr - n, 0.0);
    *hllp += maxnzr - n;
}

static inline int get_hacks_num(int num_rows, int hack_size)
{
    int hacks_num = num_rows / hack_size;
    if (num_rows % hack_size)
    {
        hacks_num++;
    }
    return hacks_num;
}

void add_null_row(matrix_format *mm, HLL_matrix *hll, int *hllp, int maxnzr)
{
    ifill(hll->col_index, *hllp, maxnzr, 0);
    dfill(hll->data, *hllp, maxnzr, 0.0);
    *hllp += maxnzr;
}

// Function to create a matrix in HLL format
void read_HLL_matrix(HLL_matrix *hll, int hack_size, matrix_format *matrix)
{
    hll->M = matrix->M;
    hll->N = matrix->N;
    hll->hack_size = hack_size;
    int *nzr = malloc(matrix->M * sizeof(int));
    if (nzr == NULL)
    {
        printf("Errore in malloc in read hll matrix\n");
        exit(EXIT_FAILURE);
    }
    get_nonzeros(matrix, nzr);

    if (hack_size > hll->M)
    {
        hack_size = hll->M;
        hll->hack_size = hack_size;
    }

    hll->hacks_num = get_hacks_num(hll->M, hack_size);
    hll->offsets = malloc((hll->hacks_num + 1) * sizeof(int));
    if (hll->offsets == NULL)
    {
        printf("Errore in malloc per hll offset in read hll\n");
        free(nzr);
        exit(EXIT_FAILURE);
    }

    int size = get_data_size(nzr, hack_size, matrix->M);

    hll->data = malloc(size * sizeof(double));
    if (hll->data == NULL)
    {
        printf("Errore in malloc per hll data\n");
        free(nzr);
        free(hll->offsets);
        hll->offsets = NULL;
        exit(EXIT_FAILURE);
    }

    hll->col_index = malloc(size * sizeof(int));
    if (hll->col_index == NULL)
    {
        printf("Errre in malloc per hll index\n");
        free(nzr);
        free(hll->offsets);
        free(hll->data);
        hll->offsets = NULL;
        hll->data = NULL;
        exit(EXIT_FAILURE);
    }

    hll->max_nzr = malloc(sizeof(int) * hll->hacks_num);
    if (hll->max_nzr == NULL)
    {
        printf("Errore in malloc per hll max_nrz\n");
        free(nzr);
        free(hll->offsets);
        free(hll->data);
        free(hll->col_index);
        hll->offsets = NULL;
        hll->data = NULL;
        hll->col_index = NULL;
        exit(EXIT_FAILURE);
    }

    int maxnzrp = 0;
    int hllp = 0;
    int offp = 0;
    int mmp = 0;
    int lo = 0;
    hll->offsets[offp++] = 0;
    int hacks_num = hll->hacks_num;
    while (hacks_num-- > 0)
    {
        if (hacks_num == 0)
        {
            hack_size = matrix->M - lo;
        }

        int maxnzr = max(nzr, lo, lo + hack_size);
        if (maxnzr > 0)
        {
            for (int i = 0; i < hack_size; ++i)
            {
                if (nzr[lo + i])
                {
                    read_row(matrix, &mmp, hll, &hllp, maxnzr);
                }
                else
                {
                    add_null_row(matrix, hll, &hllp, maxnzr);
                }
            }
        }
        hll->max_nzr[maxnzrp++] = maxnzr;
        hll->offsets[offp++] = hllp;
        lo += hack_size;
    }

    hll->offsets_num = offp;
    if (hllp != size)
    {
        printf("elements read mismatch with elements expected\n");
        free(nzr);
        free(hll->offsets);
        free(hll->data);
        free(hll->col_index);
        free(hll->max_nzr);
        nzr = NULL;
        hll->offsets = NULL;
        hll->data = NULL;
        hll->col_index = NULL;
        hll->max_nzr = NULL;
        exit(EXIT_FAILURE);
    }
    else
    {
        hll->data_num = hllp;
    }
}

// Function to destroy ellpack_matrix
void destroy_HLL_matrix(HLL_matrix *matrix)
{
    free(matrix->offsets);
    free(matrix->col_index);
    free(matrix->data);
    free(matrix->max_nzr);
}
