

#include "mpi_advance.h"
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>


#include "mpi_advance.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_i = 20;
    int max_s = pow(2, max_i);
    int s = max_s / (num_procs * num_procs);
    int n_iter = 10000;

    std::vector<double> send_data(s * num_procs);
    std::vector<double> recv_data(s * num_procs);
    std::vector<double> validation_recv_data(s * num_procs);

    //  send_data
    for (int i = 0; i < s * num_procs; i++) {
        send_data[i] = rand();
    }

    
    std::vector<int> sendcounts(num_procs, s);
    std::vector<int> recvcounts(num_procs, s);
    std::vector<int> sdispls(num_procs);
    std::vector<int> rdispls(num_procs);
    for (int i = 0; i < num_procs; i++) {
        sdispls[i] = i * s;
        rdispls[i] = i * s;
    }

    // MPIX_Comm
    MPIX_Comm* xcomm;
   
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    //Timing for PMPI_Alltoallv
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; k++) {
        PMPI_Alltoallv(send_data.data(), sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                       validation_recv_data.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    double pmpi_tfinal = (MPI_Wtime() - t0) / n_iter;

    // reduce timings for PMPI_Alltoallv
    MPI_Reduce(&pmpi_tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("PMPI_Alltoallv Time: %e seconds\n", t0);
        printf("Message Size: %ld bytes\n", s * sizeof(double));
    }

    // Timing for alltoallv_rma
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; k++) {
        alltoallv_rma(send_data.data(), sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                      recv_data.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE, xcomm);
    }
    double rma_tfinal = (MPI_Wtime() - t0) / n_iter;

    // RMA Alltoallv Time
    MPI_Reduce(&rma_tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RMA Alltoallv Time (wrapped): %e seconds\n", t0);
        printf("Message Size: %ld bytes\n", s * sizeof(double));
    }

    // Comparing RMA results with PMPI results.
    bool valid = true;
    for (int i = 0; i < s * num_procs; i++) {
        if (fabs(recv_data[i] - validation_recv_data[i]) > 1e-10) {
            fprintf(stderr, "Validation failed at rank %d, index %d: RMA %f, PMPI %f\n",
                    rank, i, recv_data[i], validation_recv_data[i]);
            valid = false;
        }
    }

   
    // Clean up
    MPIX_Comm_free(&xcomm);
    MPI_Finalize();
    return 0;
}
