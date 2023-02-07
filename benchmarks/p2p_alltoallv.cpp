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
    std::vector<double> std_alltoallv(max_s*num_procs);
    std::vector<double> loc_alltoallv(max_s*num_procs);

    std::vector<int> send_sizes(num_procs);
    std::vector<int> send_displs(num_procs+1);
    std::vector<int> recv_sizes(num_procs);
    std::vector<int> recv_displs(num_procs+1);
    send_displs[0] = 0;
    recv_displs[0] = 0;

    MPIX_Comm* locality_comm;
    MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0) printf("Testing Size %d\n", s);

        for (int i = 0; i < num_procs; i++)
        {
            send_sizes[i] = s;
            send_displs[i+1] = send_displs[i] + s;
            recv_sizes[i] = s;
            recv_displs[i+1] = recv_displs[i] + s;
        }

        for (int j = 0; j < s*num_procs; j++)
            local_data[j] = rand();

        PMPI_Alltoallv(local_data.data(),
                send_sizes.data(),
                send_displs.data(),
                MPI_DOUBLE,
                std_alltoallv.data(),
                recv_sizes.data(),
                recv_displs.data(),
                MPI_DOUBLE,
                MPI_COMM_WORLD);

        MPIX_Alltoallv(local_data.data(),
                send_sizes.data(),
                send_displs.data(),
                MPI_DOUBLE,
                loc_alltoallv.data(),
                recv_sizes.data(),
                recv_displs.data(),
                MPI_DOUBLE,
                locality_comm);

        for (int j = 0; j < s; j++)
	{
            if (fabs(std_alltoallv[j] - loc_alltoallv[j]) > 1e-10)
            {
                fprintf(stderr, 
                        "Rank %d, idx %d, std %f, loc %f\n", 
                         rank, j, std_alltoallv[j], loc_alltoallv[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            loc_alltoallv[j] = 0;
        }

        // Time Standard Alltoallv
        //
        PMPI_Alltoallv(local_data.data(),
                send_sizes.data(),
                send_displs.data(),
                MPI_DOUBLE,
                std_alltoallv.data(),
                recv_sizes.data(),
                recv_displs.data(),
                MPI_DOUBLE,
                MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            PMPI_Alltoallv(local_data.data(),
                    send_sizes.data(),
                    send_displs.data(),
                    MPI_DOUBLE,
                    std_alltoallv.data(),
                    recv_sizes.data(),
                    recv_displs.data(),
                    MPI_DOUBLE,
                    MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("PMPI_Alltoallv Time %e\n", t0);


        // Time Loc Alltoallv
        MPIX_Alltoallv(local_data.data(),
                send_sizes.data(),
                send_displs.data(),
                MPI_DOUBLE,
                loc_alltoallv.data(),
                recv_sizes.data(),
                recv_displs.data(),
                MPI_DOUBLE,
                locality_comm);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            MPIX_Alltoallv(local_data.data(),
                    send_sizes.data(),
                    send_displs.data(),
                    MPI_DOUBLE,
                    loc_alltoallv.data(),
                    recv_sizes.data(),
                    recv_displs.data(),
                    MPI_DOUBLE,
                    locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("MPIX_Alltoallv Time %e\n", t0);

    }

    MPIX_Comm_free(locality_comm);

    MPI_Finalize();
    return 0;
}
