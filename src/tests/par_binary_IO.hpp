#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#define PETSC_MAT_CODE 1211216

#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "sparse_mat.hpp"
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>

#include <stdio.h>
#include "limits.h"

bool little_endian()
{
    int num = 1;
    return (*(char *)&num == 1);
}

template <class T>
void endian_swap(T *objp)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
  std::reverse(memp, memp + sizeof(T));
}

template <class T>
void endian_swap(T* list, int n_obj)
{
    for (int i = 0; i < n_obj; i++)
        endian_swap(&(list[i]));
}

template <typename U>
void readParMatrix(const char* filename, ParMat<U>& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int64_t pos;
    int32_t header[4];
    int32_t global_nnz;
    int32_t idx;
    int n_items_read;
    double val;

    int extra;
    bool is_little_endian = false;

    int ctr, size;

    int sizeof_dbl = sizeof(val);
    int sizeof_int32 = sizeof(idx);

    FILE* ifile = fopen(filename, "rb");
    if (fseek(ifile, 0, SEEK_SET)) printf("Error seeking beginning of file\n"); 
    
    n_items_read = fread(header, sizeof_int32, 4, ifile);
    if (n_items_read == EOF) printf("EOF reading code\n");
    if (ferror(ifile)) printf("Error reading code\n");
    if (header[0] != PETSC_MAT_CODE)
    {
        is_little_endian = true;
	endian_swap(header, 4);
    }
    A.global_rows = header[1];
    A.global_cols = header[2];
    global_nnz = header[3];

    if (A.global_rows < num_procs || A.global_cols < num_procs)
    {
        if (A.global_rows < A.global_cols)
        {
            A.local_rows = 0;
            extra = A.global_rows;
            if (extra > rank)
            {
                A.local_rows = 1;
                A.first_row = rank;
            } 
            else A.first_row = extra;

            if (A.local_rows)
            {
                A.local_cols = A.global_cols / extra;
                extra = A.global_cols % extra;
                A.first_col = A.local_cols * rank;
                if (extra > rank)
                {
                    A.local_cols++;
                    A.first_col += rank;
                }
                else A.first_col += extra;
            }
            else
            {
                A.local_cols = 0;
                A.first_col = A.global_cols;
            }
        }
        else
        {
            A.local_cols = 0;
            extra = A.global_cols;
            if (extra > rank)
            {
                A.local_cols = 1;
                A.first_col = rank;
            }
            else A.first_col = extra;

            if (A.local_cols)
            {
                A.local_rows = A.global_rows / extra;
                extra = A.global_rows % extra;
                A.first_row = A.local_rows * rank;
                if (extra > rank)
                {
                    A.local_rows++;
                    A.first_row += rank;
                }
                else A.first_row += extra;
            }
            else
            {
                A.local_rows = 0;
                A.first_row = A.global_rows;
            }
        }
    }
    else
    {
        A.local_rows = A.global_rows / num_procs;
        extra = A.global_rows % num_procs;
        A.first_row = A.local_rows * rank;
        if (extra > rank)
        {
            A.local_rows++;
            A.first_row += rank;
        }
        else A.first_row += extra;

        A.local_cols = A.global_cols / num_procs;
        extra = A.global_cols % num_procs;
        A.first_col = A.local_cols * rank;
        if (extra > rank)
        {
            A.local_cols++;
            A.first_col += rank;
        }
        else A.first_col += extra;
    }

    A.on_proc.n_rows = A.local_rows;
    A.on_proc.n_cols = A.local_cols;
    A.off_proc.n_rows = A.local_rows;

    if (A.local_rows == 0)
    {
        A.off_proc_num_cols = 0;
        fclose(ifile);
	return;
    }

    std::vector<int32_t> row_sizes(A.local_rows);
    std::vector<int32_t> col_indices;
    std::vector<double> vals;
    std::vector<int> proc_nnz(num_procs);
    long nnz = 0;

    pos = (4 + A.first_row) * sizeof_int32;
    if (fseek(ifile, pos, SEEK_SET)) printf("Error seeking pos\n"); 
    n_items_read = fread(row_sizes.data(), sizeof_int32, A.local_rows, ifile);
    if (n_items_read == EOF) printf("EOF reading row sizes\n");
    if (ferror(ifile)) printf("Error reading row sizes\n");
    if (is_little_endian)
        endian_swap(row_sizes.data(), A.local_rows);
    for (int i = 0; i < A.local_rows; i++)
        nnz += row_sizes[i];

    long first_nnz = 0;
    MPI_Exscan(&nnz, &first_nnz, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (nnz)
    {
        col_indices.resize(nnz);
        vals.resize(nnz);
    }

    pos = (4 + A.global_rows + first_nnz) * sizeof_int32;
    if (fseek(ifile, pos, SEEK_SET)) printf("Error seeking pos\n"); 
    n_items_read = fread(col_indices.data(), sizeof_int32, nnz, ifile);
    if (n_items_read == EOF) printf("EOF reading col idx\n");
    if (ferror(ifile)) printf("Error reading col idx\n");
    if (is_little_endian)
        endian_swap(col_indices.data(), nnz);

    pos = (4 + A.global_rows + global_nnz) * sizeof_int32 + (first_nnz * sizeof_dbl);
    if (fseek(ifile, pos, SEEK_SET)) printf("Error seeking pos\n"); 
    n_items_read = fread(vals.data(), sizeof_dbl, nnz, ifile);
    if (n_items_read == EOF) printf("EOF reading vals\n");
    if (ferror(ifile)) printf("Error reading vals\n");
    if (is_little_endian)
        endian_swap(vals.data(), nnz);

    fclose(ifile);

    int last_col = A.first_col + A.local_cols - 1;
    A.on_proc.rowptr.resize(A.local_rows+1);
    A.off_proc.rowptr.resize(A.local_rows+1);
    A.on_proc.rowptr[0] = 0;
    A.off_proc.rowptr[0] = 0;
    ctr = 0;
    for (int i = 0; i < A.local_rows; i++)
    {
        size = row_sizes[i];
        for (int j = 0; j < size; j++)
        {
            idx = col_indices[ctr];
            val = vals[ctr++];
            if ((int)idx >= A.first_col && idx <= last_col)
            {
                A.on_proc.col_idx.push_back(idx - A.first_col);
                A.on_proc.data.push_back(val);
            }
            else
            {
                A.off_proc.col_idx.push_back(idx);
                A.off_proc.data.push_back(val);
            }
        }
        A.on_proc.rowptr[i+1] = A.on_proc.col_idx.size();
        A.off_proc.rowptr[i+1] = A.off_proc.col_idx.size();
    }
    A.on_proc.nnz = A.on_proc.col_idx.size();
    A.off_proc.nnz = A.off_proc.col_idx.size();

    std::map<long, int> orig_to_new;
    std::copy(A.off_proc.col_idx.begin(), A.off_proc.col_idx.end(),
            std::back_inserter(A.off_proc_columns));
    std::sort(A.off_proc_columns.begin(), A.off_proc_columns.end());

    int prev_col = -1;
    A.off_proc_num_cols = 0;
    for (std::vector<long>::iterator it = A.off_proc_columns.begin(); 
            it != A.off_proc_columns.end(); ++it)
    {
        if (*it != prev_col)
        {
            orig_to_new[*it] = A.off_proc_num_cols;
            A.off_proc_columns[A.off_proc_num_cols++] = *it;
            prev_col = *it;
        }
    }
    if (A.off_proc_num_cols)
        A.off_proc_columns.resize(A.off_proc_num_cols);

    for (std::vector<int>::iterator it = A.off_proc.col_idx.begin();
            it != A.off_proc.col_idx.end(); ++it)
    {
        *it = orig_to_new[*it];
    }

    A.off_proc.n_cols = A.off_proc_num_cols;
}

#endif

