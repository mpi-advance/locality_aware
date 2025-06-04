#include "mpi_advance.h"
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_i = 20;
    int max_s = pow(2, max_i);
    int s = max_s / num_procs;  // Adjusted for Alltoall removed all displacements for send and recv)
    int n_iter = 10000;

    std::vector<double> send_data(s * num_procs);
    std::vector<double> recv_data(s * num_procs);
    std::vector<double> validation_recv_data(s * num_procs);

    // Initializing send_data
    for (int i = 0; i < s * num_procs; i++) {
        send_data[i] = rand();
    }

    
    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    // Timing for PMPI_Alltoall
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; k++) {
        PMPI_Alltoall(send_data.data(), s, MPI_DOUBLE, 
                      validation_recv_data.data(), s, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    double pmpi_tfinal = (MPI_Wtime() - t0) / n_iter;

    // Reducing timings for PMPI_Alltoall
    MPI_Reduce(&pmpi_tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("PMPI_Alltoall Time: %e seconds\n", t0);
        printf("Message Size: %ld bytes\n", s * sizeof(double));
    }

    // Timing for alltoall_rma
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; k++) {
        alltoall_rma(send_data.data(), s, MPI_DOUBLE,
                     recv_data.data(), s, MPI_DOUBLE, xcomm);
    }
    double rma_tfinal = (MPI_Wtime() - t0) / n_iter;

    // Reducing timings for RMA Alltoall
    MPI_Reduce(&rma_tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RMA Alltoall Time: %e seconds\n", t0);
        printf("Message Size: %ld bytes\n", s * sizeof(double));
    }

    // Comparing RMA results with PMPI results
    bool valid = true;
    for (int i = 0; i < s * num_procs; i++) {
        if (fabs(recv_data[i] - validation_recv_data[i]) > 1e-10) {
            fprintf(stderr, "Validation failed at rank %d, index %d: RMA %f, PMPI %f\n",
                    rank, i, recv_data[i], validation_recv_data[i]);
            valid = false;
        }
    }

    
    MPIX_Comm_free(&xcomm);
    MPI_Finalize();
    return 0;
}
