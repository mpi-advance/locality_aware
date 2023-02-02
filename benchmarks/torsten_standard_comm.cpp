#include "mpi_advance.h"
#include "tests/sparse_mat.hpp"
#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for(int i = 0; i < argc; i++) 
    {
        printf("Arg %d:%s\n", i, argv[i]);
    }

    test_matrix("test",TORSTEN);
    MPI_Finalize();
    return 0;
}
