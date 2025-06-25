#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void compare_alltoall_results(std::vector<int>& pmpi, std::vector<int>& mpix, int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int j = 0; j < s*num_procs; j++)
    {
        if (pmpi[j] != mpix[j])
        {
            fprintf(stderr, "MPIX Alltoall != PMPI, position %d, pmpi %d, mpix %d\n", 
                    j, pmpi[j], mpix[j]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Test Integer Alltoall
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s*num_procs);

    std::vector<int> pmpi_alltoall(max_s*num_procs);
    std::vector<int> mpix_alltoall(max_s*num_procs);

    MPIX_Comm* locality_comm;
    MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);
    update_locality(locality_comm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int j = 0; j < num_procs; j++)
            for (int k = 0; k < s; k++)
                local_data[j*s + k] = rank*10000 + j*100 + k;

        // Standard Alltoall
        PMPI_Alltoall(local_data.data(), 
                s,
                MPI_INT, 
                pmpi_alltoall.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);

        // Locality-Aware Pairwise Alltoall
        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        MPIX_Alltoall(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);


        // Test Standard Pairwise
        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_pairwise(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);

        // Test Standard Nonblocking
        alltoall_nonblocking(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);


        // Test Hierarchical + Pairwise
        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_hierarchical_pairwise(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);

        // Test Hierarchical + Nonblocking 
        alltoall_hierarchical_nonblocking(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);


        // Test Multileader + Pairwise
        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_multileader_pairwise(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);

        // Test Multileader + Nonblocking 
        alltoall_multileader_nonblocking(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);


        // Test Node Aware + Pairwise
        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_node_aware_pairwise(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);

        // Test Node Aware + Nonblocking 
        alltoall_node_aware_nonblocking(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);

        // Test Locality Aware + Pairwise
        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_locality_aware_pairwise(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);

        // Test Locality Aware + Nonblocking 
        alltoall_locality_aware_nonblocking(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);


        // Test Multileader + Locality Aware + Pairwise
        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        alltoall_multileader_locality_pairwise(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);

        // Test Multileader + Locality Aware + Nonblocking 
        alltoall_multileader_locality_nonblocking(local_data.data(), 
                s, 
                MPI_INT,
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);


    }

    MPIX_Comm_free(&locality_comm);

    MPI_Finalize();
    return 0;
} // end of main() //


