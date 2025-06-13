#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_i = 15;
    int max_s = pow(2, max_i);
    int n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));
    std::vector<double> local_data(max_s*num_procs);
    std::vector<double> std_alltoall(max_s*num_procs);
    std::vector<double> loc_alltoall(max_s*num_procs);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0) printf("Testing Size %d\n", s);

        for (int j = 0; j < s*num_procs; j++)
            local_data[j] = rand();

        PMPI_Alltoall(local_data.data(),
                s,
                MPI_DOUBLE,
                std_alltoall.data(),
                s,
                MPI_DOUBLE,
                MPI_COMM_WORLD);

        MPIX_Alltoall(local_data.data(),
                s,
                MPI_DOUBLE,
                loc_alltoall.data(),
                s,
                MPI_DOUBLE,
                xcomm);

        for (int j = 0; j < s; j++)
	{
            if (fabs(std_alltoall[j] - loc_alltoall[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, std %d, loc %d\n", 
                         rank, j, std_alltoall[j], loc_alltoall[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            loc_alltoall[j] = 0;
        }

        // Time Standard Alltoall
        //
        PMPI_Alltoall(local_data.data(),
                s,
                MPI_DOUBLE,
                std_alltoall.data(),
                s,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            PMPI_Alltoall(local_data.data(),
                    s,
                    MPI_DOUBLE,
                    std_alltoall.data(),
                    s,
                    MPI_DOUBLE,
                    MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("PMPI_Alltoall Time %e\n", t0);


        // Time Loc Alltoall
        MPIX_Alltoall(local_data.data(),
                s,
                MPI_DOUBLE,
                loc_alltoall.data(),
                s,
                MPI_DOUBLE,
                xcomm);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            MPIX_Alltoall(local_data.data(),
                    s,
                    MPI_DOUBLE,
                    loc_alltoall.data(),
                    s,
                    MPI_DOUBLE,
                    xcomm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX_Alltoall Time %e\n", t0);

    }

    MPIX_Comm_free(&xcomm);

    MPI_Finalize();
    return 0;
}
