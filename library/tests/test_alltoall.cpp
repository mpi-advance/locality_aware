#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

#include "locality_aware.h"
//#include "collective/alltoall.h"

void compare_alltoall_results(std::vector<int>& pmpi, std::vector<int>& mpil, int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int j = 0; j < s * num_procs; j++)
    {
        if (pmpi[j] != mpil[j])
        {
            fprintf(stderr,
                    "MPIL Alltoall != PMPI, position %d, pmpi %d, mpil %d\n",
                    j,
                    pmpi[j],
                    mpil[j]);
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
    std::vector<int> local_data(max_s * num_procs);

    std::vector<int> pmpi_alltoall(max_s * num_procs);
    std::vector<int> mpil_alltoall(max_s * num_procs);

    MPIL_Comm* locality_comm;
    MPIL_Comm_init(&locality_comm, MPI_COMM_WORLD);
    MPIL_Comm_update_locality(locality_comm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int j = 0; j < num_procs; j++)
        {
            for (int k = 0; k < s; k++)
            {
                local_data[j * s + k] = rank * 10000 + j * 100 + k;
            }
        }

        // Standard Alltoall
        PMPI_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      pmpi_alltoall.data(),
                      s,
                      MPI_INT,
                      MPI_COMM_WORLD);

        // Locality-Aware Pairwise Alltoall
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
        MPIL_Alltoall(local_data.data(),
                      s,
                      MPI_INT,
                      mpil_alltoall.data(),
                      s,
                      MPI_INT,
                      locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Standard Pairwise
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
		MPIL_Set_alltoall_algorithm(ALLTOALL_PAIRWISE);
        MPIL_Alltoall(local_data.data(),
					  s,
					  MPI_INT,
					  mpil_alltoall.data(),
					  s,
					  MPI_INT,
					  locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Standard Nonblocking
		MPIL_Set_alltoall_algorithm(ALLTOALL_NONBLOCKING);
        MPIL_Alltoall(local_data.data(),
					 s,
					 MPI_INT,
					 mpil_alltoall.data(),
					 s,
					 MPI_INT,
					 locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Hierarchical + Pairwise
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
		MPIL_Set_alltoall_algorithm(ALLTOALL_HIERARCHICAL_PAIRWISE);
        MPIL_Alltoall(local_data.data(),
					   s,
					   MPI_INT,
					   mpil_alltoall.data(),
					   s,
					   MPI_INT,
					   locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Hierarchical + Nonblocking
		MPIL_Set_alltoall_algorithm(ALLTOALL_HIERARCHICAL_NONBLOCKING);
        MPIL_Alltoall(local_data.data(),
					  s,
					  MPI_INT,
					  mpil_alltoall.data(),
					  s,
					  MPI_INT,
					  locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Multileader + Pairwise
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
		MPIL_Set_alltoall_algorithm(ALLTOALL_MULTILEADER_PAIRWISE);
        MPIL_Alltoall(local_data.data(),
					  s,
					  MPI_INT,
					  mpil_alltoall.data(),
					  s,
					  MPI_INT,
					  locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Multileader + Nonblocking
		MPIL_Set_alltoall_algorithm(ALLTOALL_MULTILEADER_NONBLOCKING);
        MPIL_Alltoall(local_data.data(),
						 s,
						 MPI_INT,
						 mpil_alltoall.data(),
						 s,
						 MPI_INT,
						 locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Node Aware + Pairwise
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
		MPIL_Set_alltoall_algorithm(ALLTOALL_NODE_AWARE_PAIRWISE);
        MPIL_Alltoall(local_data.data(),
					 s,
					 MPI_INT,
					 mpil_alltoall.data(),
					 s,
					 MPI_INT,
					 locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Node Aware + Nonblocking
		MPIL_Set_alltoall_algorithm(ALLTOALL_NODE_AWARE_NONBLOCKING);
        MPIL_Alltoall(local_data.data(),
						s,
						MPI_INT,
						mpil_alltoall.data(),
						s,
						MPI_INT,
						locality_comm);
		compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Locality Aware + Pairwise
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
		MPIL_Set_alltoall_algorithm(ALLTOALL_LOCALITY_AWARE_PAIRWISE);
        MPIL_Alltoall(local_data.data(),
					 s,
					 MPI_INT,
					 mpil_alltoall.data(),
					 s,
					 MPI_INT,
					 locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Locality Aware + Nonblocking
		MPIL_Set_alltoall_algorithm(ALLTOALL_LOCALITY_AWARE_NONBLOCKING);
       MPIL_Alltoall(local_data.data(),
					s,
					MPI_INT,
					mpil_alltoall.data(),
					s,
					MPI_INT,
					locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Multileader + Locality Aware + Pairwise
        std::fill(mpil_alltoall.begin(), mpil_alltoall.end(), 0);
		MPIL_Set_alltoall_algorithm(ALLTOALL_MULTILEADER_LOCALITY_PAIRWISE);
        MPIL_Alltoall(local_data.data(),
					   s,
					   MPI_INT,
					   mpil_alltoall.data(),
					   s,
					   MPI_INT,
					   locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);

        // Test Multileader + Locality Aware + Nonblocking
		MPIL_Set_alltoall_algorithm(ALLTOALL_MULTILEADER_LOCALITY_NONBLOCKING);
		
        MPIL_Alltoall(local_data.data(),
					  s,
					  MPI_INT,
					  mpil_alltoall.data(),
					  s,
					  MPI_INT,
					  locality_comm);
        compare_alltoall_results(pmpi_alltoall, mpil_alltoall, s);
    }

    MPIL_Comm_free(&locality_comm);

    MPI_Finalize();
    return 0;
}  // end of main() //
