#include "mpi_advance.h"
#include "tests/sparse_mat.hpp"
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <tuple>

std::tuple<double, int, int> test_matrix(const char* filename, COMM_ALGORITHM algorithm);
/*
 * Argument format: 
 * argv[0] - default program name
 * argv[1] - name of matrix file
 * argv[2] - number of tests to run
 * argv[3] - name of matrix (simplified)
 * argv[4] - test to run (STANDARD vs. TORSTEN 
*/
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    int num_tests = std::stoi(argv[2],nullptr);
    COMM_ALGORITHM algo; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if(strcmp("TORSTEN", argv[4]) == 0) { algo = TORSTEN; }
    else if(strcmp("STANDARD", argv[4]) == 0) { algo = STANDARD; }
    else if(strcmp("RMA", argv[4]) == 0) {algo == RMA; }
    else {
        if(rank == 0)
        {
            fprintf(stderr, "choose between STANDARD or THORSTEN or RMA, exiting");
        }
        MPI_Finalize();
        return 1;
    }

    /* Print information about tests */
    if(rank == 0) 
    {
        if(algo == STANDARD) { printf("STANDARD, %d PROCS, %d TESTS, %s\n", num_procs, num_tests, argv[3]); }
        else if(algo == TORSTEN) { printf("TORSTEN, %d PROCS, %d TESTS, %s\n", num_procs, num_tests, argv[3]); }
        else if(algo == RMA) { printf("RMA, %d PROCS, %d TESTS, %s\n", num_procs, num_tests, argv[3])}
    }

    /* Run num_tests number of tests and print info about message sizes / time taken*/
    for(int i = 0; i < num_tests; i++) 
    {
        MPI_Barrier(MPI_COMM_WORLD);
        std::tuple<double, int, int> info = test_matrix(argv[1], algo);
        double max_time = 0;
        int max_msg_count = 0;
        int max_msg_size = 0;
        MPI_Allreduce(&(std::get<0>(info)), &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&(std::get<1>(info)), &max_msg_count, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&(std::get<2>(info)), &max_msg_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(rank == 0 && i == 0) 
        {
            printf("MAX_MSG_COUNT %d, MAX_MSG_SIZE %d\n",max_msg_count, max_msg_size);
        }
        if(rank == 0) 
        {
            printf("%lf\n", max_time);
        }
    }

    MPI_Finalize();
    return 0;
}
