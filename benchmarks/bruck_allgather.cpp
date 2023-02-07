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

    MPIX_Comm* locality_comm;
    MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);

    int max_i = 5;
    int max_s = pow(2, max_i);
    int n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));
    std::vector<double> local_data(max_s);

    std::vector<double> std_allgather(max_s*num_procs);
    std::vector<double> bruck_allgather(max_s*num_procs);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0) printf("Testing Size %d\n", s);

        for (int j = 0; j < s; j++)
            local_data[j] = rand();

        PMPI_Allgather(local_data.data(), 
                s,
                MPI_DOUBLE, 
                std_allgather.data(), 
                s, 
                MPI_DOUBLE,
                MPI_COMM_WORLD);

        allgather_bruck(local_data.data(), 
                s, 
                MPI_DOUBLE,
                bruck_allgather.data(), 
                s, 
                MPI_DOUBLE,
                MPI_COMM_WORLD);

        for (int j = 0; j < s; j++)
	{
            if (fabs(std_allgather[j] - bruck_allgather[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, std %f, bruck %f\n", 
                         rank, j, std_allgather[j], bruck_allgather[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            bruck_allgather[j] = 0;
        }

        allgather_loc_bruck(local_data.data(),
                s,
                MPI_DOUBLE,
                bruck_allgather.data(),
                s,
                MPI_DOUBLE,
                locality_comm);

        for (int j = 0; j < s; j++)
        {
            if (fabs(std_allgather[j] - bruck_allgather[j]) > 1e-10)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, std %f, bruck %f\n",
                         rank, j, std_allgather[j], bruck_allgather[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            bruck_allgather[j] = 0;
        }

        allgather_hier_bruck(local_data.data(),
                s,
                MPI_DOUBLE,
                bruck_allgather.data(),
                s,
                MPI_DOUBLE,
                locality_comm);

        for (int j = 0; j < s; j++)
        {
            if (fabs(std_allgather[j] - bruck_allgather[j]) > 1e-10)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, std %f, bruck %f\n",
                         rank, j, std_allgather[j], bruck_allgather[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            bruck_allgather[j] = 0;
        }

        allgather_mult_hier_bruck(local_data.data(),
                s,
                MPI_DOUBLE,
                bruck_allgather.data(),
                s,
                MPI_DOUBLE,
                locality_comm);

        for (int j = 0; j < s; j++)
            if (fabs(std_allgather[j] - bruck_allgather[j]) > 1e-10)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, std %f, bruck %f\n",
                         rank, j, std_allgather[j], bruck_allgather[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }


        // Time Standard Allgather
        PMPI_Allgather(local_data.data(), 
            s,
            MPI_DOUBLE, 
            std_allgather.data(), 
            s, 
            MPI_DOUBLE,
            MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            PMPI_Allgather(local_data.data(), 
                    s,
                    MPI_DOUBLE, 
                    std_allgather.data(), 
                    s, 
                    MPI_DOUBLE,
                    MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("PMPI_Allgather Time %e\n", t0);


        // Time Bruck Allgather
        allgather_bruck(local_data.data(),
            s,
            MPI_DOUBLE, 
            bruck_allgather.data(), 
            s, 
            MPI_DOUBLE,
            MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            allgather_bruck(local_data.data(), 
                    s,
                    MPI_DOUBLE, 
                    bruck_allgather.data(), 
                    s, 
                    MPI_DOUBLE,
                    MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("allgather_bruck Time %e\n", t0);

        // Time Locality-Aware Bruck Allgather
        allgather_loc_bruck(local_data.data(),
            s,
            MPI_DOUBLE,
            bruck_allgather.data(),
            s,
            MPI_DOUBLE,
            locality_comm);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {   
            allgather_loc_bruck(local_data.data(),
                    s,
                    MPI_DOUBLE, 
                    bruck_allgather.data(),
                    s, 
                    MPI_DOUBLE,
                    locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("allgather_loc_bruck Time %e\n", t0);


        // Time Hierarchical (1PPN) Bruck Allgather
        allgather_hier_bruck(local_data.data(),
            s,
            MPI_DOUBLE,
            bruck_allgather.data(),
            s,
            MPI_DOUBLE,
            locality_comm);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {   
            allgather_hier_bruck(local_data.data(),
                    s,
                    MPI_DOUBLE, 
                    bruck_allgather.data(),
                    s, 
                    MPI_DOUBLE,
                    locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("allgather_hier_bruck Time %e\n", t0);

        // Time Hierarchical (All PPN) Bruck Allgather
        allgather_mult_hier_bruck(local_data.data(),
            s,
            MPI_DOUBLE,
            bruck_allgather.data(),
            s,
            MPI_DOUBLE,
            locality_comm);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {   
            allgather_mult_hier_bruck(local_data.data(),
                    s,
                    MPI_DOUBLE, 
                    bruck_allgather.data(),
                    s, 
                    MPI_DOUBLE,
                    locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("allgather_mult_hier_bruck Time %e\n", t0);
    }

    MPI_Finalize();
    return 0;
}
