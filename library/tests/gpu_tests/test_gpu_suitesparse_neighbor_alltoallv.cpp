#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "locality_aware.h"
#include "gpu_utils.h"

#include "common.hpp"
#include "par_binary_IO.hpp"
#include "sparse_mat.hpp"

void compare_neighbor_alltoallv_results(std::vector<int>& pmpi_recv_vals,
                                        std::vector<int>& mpix_recv_vals,
                                        int s)
{
    for (int i = 0; i < s; i++)
    {
        if (pmpi_recv_vals[i] != mpix_recv_vals[i])
        {
            fprintf(stderr,
                    "PMPI recv != MPIL: position %d, pmpi %d, mpix %d\n",
                    i,
                    pmpi_recv_vals[i],
                    mpix_recv_vals[i]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void test_matrix(const char* filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<int> A;
    int idx;
    readParMatrix(filename, A);
    form_comm(A);

    std::vector<int> send_vals, alltoallv_send_vals;
    std::vector<int> pmpi;
    std::vector<int> mpil;
    std::vector<int> device_data;
    std::vector<long> send_indices;
    void* gpu_sendbuf = NULL;
    void* gpu_recvbuf = NULL;

    int ierr;

    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols);
        std::iota(send_vals.begin(), send_vals.end(), 0);
        for (int i = 0; i < A.on_proc.n_cols; i++)
        {
            send_vals[i] += (rank * 1000);
        }
    }

    if (A.recv_comm.size_msgs)
    {
        pmpi.resize(A.recv_comm.size_msgs);
        mpil.resize(A.recv_comm.size_msgs);
        device_data.resize(A.recv_comm.size_msgs);
        ierr = gpuMalloc(&gpu_recvbuf, A.recv_comm.size_msgs * sizeof(int));
        gpu_check(ierr);
    }

    if (A.send_comm.size_msgs)
    {
        alltoallv_send_vals.resize(A.send_comm.size_msgs);
        send_indices.resize(A.send_comm.size_msgs);
        ierr = gpuMalloc(&gpu_sendbuf, A.send_comm.size_msgs * sizeof(int));
        gpu_check(ierr);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
        {
            idx                    = A.send_comm.idx[i];
            alltoallv_send_vals[i] = send_vals[idx];
            send_indices[i]        = A.send_comm.idx[i] + A.first_col;
        }
        ierr = gpuMemcpy(gpu_sendbuf, alltoallv_send_vals.data(), 
                A.send_comm.size_msgs * sizeof(int),
                gpuMemcpyHostToDevice);
        gpu_check(ierr);
    }

    communicate(A, send_vals, mpil, MPI_INT);

    MPI_Comm std_comm;
    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Info* xinfo;
    MPIL_Info_init(&xinfo);

    MPIL_Topo* topo;
    MPIL_Topo_init(A.recv_comm.n_msgs,
                   A.recv_comm.procs.data(),
                   A.recv_comm.counts.data(),
                   A.send_comm.n_msgs,
                   A.send_comm.procs.data(),
                   A.send_comm.counts.data(),
                   xinfo,
                   &topo);

    int* s = A.recv_comm.procs.data();
    if (A.recv_comm.n_msgs == 0)
    {
        s = MPI_WEIGHTS_EMPTY;
    }
    int* d = A.send_comm.procs.data();
    if (A.send_comm.n_msgs == 0)
    {
        d = MPI_WEIGHTS_EMPTY;
    }

    PMPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                    A.recv_comm.n_msgs,
                                    s,
                                    MPI_UNWEIGHTED,
                                    A.send_comm.n_msgs,
                                    d,
                                    MPI_UNWEIGHTED,
                                    MPI_INFO_NULL,
                                    0,
                                    &std_comm);

    // Test PMPI on CPU.  Need non-NULL send_counts for some
    // versions of MPI
    int* send_counts = A.send_comm.counts.data();
    if (A.send_comm.counts.data() == NULL)
    {
        send_counts = new int[1];
    }
    int* recv_counts = A.recv_comm.counts.data();
    if (A.recv_comm.counts.data() == NULL)
    {
        recv_counts = new int[1];
    }
    PMPI_Neighbor_alltoallv(alltoallv_send_vals.data(),
                            send_counts,
                            A.send_comm.ptr.data(),
                            MPI_INT,
                            pmpi.data(),
                            recv_counts,
                            A.recv_comm.ptr.data(),
                            MPI_INT,
                            std_comm);
    if (A.send_comm.counts.data() == NULL)
    {
        delete[] send_counts;
    }
    if (A.recv_comm.counts.data() == NULL)
    {
        delete[] recv_counts;
    }
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);


    // Standard MPIL_Neighbor collective on CPU
    MPIL_Set_alltoallv_neighbor_algorithm(NEIGHBOR_ALLTOALLV_STANDARD);
    std::fill(mpil.begin(), mpil.end(), 0);
    MPIL_Neighbor_alltoallv_topo(alltoallv_send_vals.data(),
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 MPI_INT,
                                 mpil.data(),
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);

#if defined(GPU_AWARE)

    // Standard MPIL_Nieghbor collective on GPU
    MPIL_Set_alltoallv_neighbor_algorithm(NEIGHBOR_ALLTOALLV_GPU_STANDARD);
    ierr = gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    gpu_check(ierr);
    MPIL_Neighbor_alltoallv_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm);
    ierr = gpuMemcpy(mpil.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    gpu_check(ierr);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);

    // Locality MPIL_Nieghbor collective on GPU
    MPIL_Set_alltoallv_neighbor_algorithm(NEIGHBOR_ALLTOALLV_GPU_LOCALITY);
    ierr = gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    gpu_check(ierr);
    MPIL_Neighbor_alltoallv_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm);
    ierr = gpuMemcpy(mpil.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    gpu_check(ierr);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);

#endif


    // Standard MPIL_Nieghbor collective on GPU
    MPIL_Set_alltoallv_neighbor_algorithm(NEIGHBOR_ALLTOALLV_CTC_STANDARD);
    ierr = gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    gpu_check(ierr);
    MPIL_Neighbor_alltoallv_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm);
    ierr = gpuMemcpy(mpil.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    gpu_check(ierr);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);

    // Locality MPIL_Nieghbor collective on GPU
    MPIL_Set_alltoallv_neighbor_algorithm(NEIGHBOR_ALLTOALLV_CTC_LOCALITY);
    ierr = gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    gpu_check(ierr);
    MPIL_Neighbor_alltoallv_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm);
    ierr = gpuMemcpy(mpil.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    gpu_check(ierr);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);

    if (A.recv_comm.size_msgs)
    {
        ierr = gpuFree(gpu_recvbuf);
        gpu_check(ierr);
    }
    if (A.send_comm.size_msgs)
    {
        ierr = gpuFree(gpu_sendbuf);
        gpu_check(ierr);
    }    

    MPIL_Topo_free(&topo);
    MPIL_Info_free(&xinfo);
    MPIL_Comm_free(&xcomm);
    PMPI_Comm_free(&std_comm);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPIL_Init(MPI_COMM_WORLD);
    test_all_matrices();
    MPIL_Finalize();
    MPI_Finalize();
    return 0;
}  // end of main() //
