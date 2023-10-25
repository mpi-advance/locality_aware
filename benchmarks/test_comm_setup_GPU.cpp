#include "mpi_advance.h"
#include "../src/tests/par_binary_IO.hpp"
#include "../src/tests/sparse_mat.hpp"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

int main(int argc, char* argv[])
{
    char* filename;
    if (argc <= 1)
    {
        printf("Need Command Line Argument for Filename!\n");
        return -1;
    }
    else
    {
        filename = argv[1]; 
    }

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal; 
    int n_iter = 3; 

    ParMat<int> A;
    readParMatrix(filename, A);
    form_recv_comm(A);
    
    A.recv_comm->idx

    MPI_Finalize();
    return 0;
}