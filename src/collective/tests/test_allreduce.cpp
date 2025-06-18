#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void compare_allreduce_results(std::vector<double>& pmpi, std::vector<double>& mpix, int s)
{
    for (int j = 0; j < s; j++)
    {
        if (fabs((pmpi[j] - mpix[j])/mpix[j]) > 1e-06)
        {
            fprintf(stderr, "MPIX Allreduce != PMPI, position %d, pmpi %e, mpix %e\n", 
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
    int max_i = 15;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<double> local_data(max_s);
    for (int i = 0; i < max_s; i++)
        local_data[i] = rand() / RAND_MAX;

    std::vector<double> pmpi_alltoall(max_s);
    std::vector<double> mpix_alltoall(max_s);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Standard Allreduce
        PMPI_Allreduce(local_data.data(), 
                pmpi_alltoall.data(), 
                s, 
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);

        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        MPIX_Allreduce(local_data.data(), 
                mpix_alltoall.data(), 
                s, 
                MPI_DOUBLE,
                MPI_SUM,
                xcomm);
        compare_allreduce_results(pmpi_alltoall, mpix_alltoall, s);

        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        allreduce_lane(local_data.data(), 
                mpix_alltoall.data(), 
                s, 
                MPI_DOUBLE,
                MPI_SUM,
                xcomm);
        compare_allreduce_results(pmpi_alltoall, mpix_alltoall, s);

        std::fill(mpix_alltoall.begin(), mpix_alltoall.end(), 0);
        allreduce_loc(local_data.data(), 
                mpix_alltoall.data(), 
                s, 
                MPI_DOUBLE,
                MPI_SUM,
                xcomm);
        compare_allreduce_results(pmpi_alltoall, mpix_alltoall, s);
    }

    MPIX_Comm_free(&xcomm);


    MPI_Finalize();
    return 0;
} // end of main() //


