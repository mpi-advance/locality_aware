#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "locality_aware.h"
#include "tests/common.hpp"
#include "tests/par_binary_IO.hpp"
#include "tests/sparse_mat.hpp"

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
        gpuMalloc(&gpu_recvbuf, A.recv_comm.size_msgs * sizeof(int));
    }

    if (A.send_comm.size_msgs)
    {
        alltoallv_send_vals.resize(A.send_comm.size_msgs);
        send_indices.resize(A.send_comm.size_msgs);
        gpuMalloc(&gpu_sendbuf, A.send_comm.size_msgs * sizeof(int));
        for (int i = 0; i < A.send_comm.size_msgs; i++)
        {
            idx                    = A.send_comm.idx[i];
            alltoallv_send_vals[i] = send_vals[idx];
            send_indices[i]        = A.send_comm.idx[i] + A.first_col;
        }
        gpuMemcpy(gpu_sendbuf, alltoallv_send_vals.data(), 
                A.send_comm.size_msgs * sizeof(int),
                gpuMemcpyHostToDevice);
    }

    communicate(A, send_vals, mpix, MPI_INT);

    MPI_Comm std_comm;
    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Info* info;
    MPIL_Info_init(&info);

    MPIL_Topo* topo;
    MPIL_Topo_init(A.recv_comm.n_msgs,
                   A.recv_comm.procs.data(),
                   A.recv_comm.counts.data(),
                   A.send_comm.n_msgs,
                   A.send_comm.procs.data(),
                   A.send_comm.counts.data(),
                   info,
                   &topo);

    MPIL_Request* xrequest;

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



    // Standard MPIL_Neighbor persistent collective on CPU
    MPIL_Set_alltoallv_neighbor_init_algorithm(NEIGHBOR_ALLTOALLV_INIT_STANDARD);
    std::fill(mpil.begin(), mpil.end(), 0);
    MPIL_Neighbor_alltoallv_init_topo(alltoallv_send_vals.data(),
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 MPI_INT,
                                 mpil.data(),
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm,
                                 info,
                                 &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, MPI_STATUS_IGNORE);
    MPIL_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);


#if defined(GPU_AWARE)
    // Standard MPIL_Nieghbor persistent collective on GPU
    MPIL_Set_alltoallv_neighbor_init_algorithm(NEIGHBOR_ALLTOALLV_INIT_GPU_STANDARD);
    gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    MPIL_Neighbor_alltoallv_init_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm,
                                 info,
                                 &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, MPI_STATUS_IGNORE);
    MPIL_Request_free(&xrequest);
    gpuMemcpy(mpix.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);

    // Locality MPIL_Nieghbor collective on GPU
    MPIL_Set_alltoallv_neighbor_init_algorithm(NEIGHBOR_ALLTOALLV_INIT_GPU_LOCALITY);
    gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    MPIL_Neighbor_alltoallv_init_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm,
                                 info,
                                 &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, MPI_STATUS_IGNORE);
    MPIL_Request_free(&xrequest);
    gpuMemcpy(mpix.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);


    // Extended: Standard MPIL_Nieghbor persistent collective on GPU
    MPIL_Set_alltoallv_neighbor_init_algorithm(NEIGHBOR_ALLTOALLV_INIT_GPU_STANDARD);
    gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    MPIL_Neighbor_alltoallv_init_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 send_indices.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 A.off_proc_columns.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm,
                                 info,
                                 &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, MPI_STATUS_IGNORE);
    MPIL_Request_free(&xrequest);
    gpuMemcpy(mpix.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);

    // Extended: Locality MPIL_Nieghbor collective on GPU
    MPIL_Set_alltoallv_neighbor_init_algorithm(NEIGHBOR_ALLTOALLV_INIT_GPU_LOCALITY);
    gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    MPIL_Neighbor_alltoallv_init_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 send_indices.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 A.off_proc_columns.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm,
                                 info,
                                 &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, MPI_STATUS_IGNORE);
    MPIL_Request_free(&xrequest);
    gpuMemcpy(mpix.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);

#endif




    // Standard MPIL_Nieghbor persistent collective on GPU
    MPIL_Set_alltoallv_neighbor_init_algorithm(NEIGHBOR_ALLTOALLV_INIT_CTC_STANDARD);
    gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    MPIL_Neighbor_alltoallv_init_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm,
                                 info,
                                 &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, MPI_STATUS_IGNORE);
    MPIL_Request_free(&xrequest);
    gpuMemcpy(mpix.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);

    // Locality MPIL_Nieghbor collective on GPU
    MPIL_Set_alltoallv_neighbor_init_algorithm(NEIGHBOR_ALLTOALLV_INIT_CTC_LOCALITY);
    gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    MPIL_Neighbor_alltoallv_init_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm,
                                 info,
                                 &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, MPI_STATUS_IGNORE);
    MPIL_Request_free(&xrequest);
    gpuMemcpy(mpix.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);


    // Extended: Standard MPIL_Nieghbor persistent collective on GPU
    MPIL_Set_alltoallv_neighbor_init_algorithm(NEIGHBOR_ALLTOALLV_INIT_CTC_STANDARD);
    gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    MPIL_Neighbor_alltoallv_init_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 send_indices.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 A.off_proc_columns.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm,
                                 info,
                                 &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, MPI_STATUS_IGNORE);
    MPIL_Request_free(&xrequest);
    gpuMemcpy(mpix.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);

    // Extended: Locality MPIL_Nieghbor collective on GPU
    MPIL_Set_alltoallv_neighbor_init_algorithm(NEIGHBOR_ALLTOALLV_INIT_CTC_LOCALITY);
    gpuMemset(gpu_recvbuf, 0, A.recv_comm.size_msgs*sizeof(int));
    MPIL_Neighbor_alltoallv_init_topo(gpu_sendbuf,
                                 A.send_comm.counts.data(),
                                 A.send_comm.ptr.data(),
                                 send_indices.data(),
                                 MPI_INT,
                                 gpu_recvbuf,
                                 A.recv_comm.counts.data(),
                                 A.recv_comm.ptr.data(),
                                 A.off_proc_columns.data(),
                                 MPI_INT,
                                 topo,
                                 xcomm,
                                 info,
                                 &xrequest);
    MPIL_Start(xrequest);
    MPIL_Wait(xrequest, MPI_STATUS_IGNORE);
    MPIL_Request_free(&xrequest);
    gpuMemcpy(mpix.data(), gpu_recvbuf, A.recv_comm.size_msgs*sizeof(int),
            gpuMemcpyDeviceToHost);
    compare_neighbor_alltoallv_results(
        pmpi, mpil, A.recv_comm.size_msgs);






    if (A.recv_comm.size_msgs)
    {
        gpuFree(&gpu_recvbuf);
    }
    if (A.send_comm.size_msgs)
    {
        gpuFree(&gpu_sendbuf);
    }    

    MPIL_Topo_free(&topo);
    MPIL_Info_free(&info);
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

