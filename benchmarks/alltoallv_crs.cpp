#include <numeric>

#include "locality_aware.h"
#include "par_binary_IO.hpp"
#include "sparse_mat.hpp"

void compare(int n_recvs,
             int s_recvs,
             int* src,
             int* counts,
             int* displs,
             long* indices,
             int orig_n_recvs,
             int orig_s_recvs,
             int* orig_proc_counts,
             int* orig_proc_displs,
             long* orig_indices)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (n_recvs != orig_n_recvs)
    {
        printf("Num Messages Incorrect! Rank %d got %d, should be %d\n",
               rank,
               n_recvs,
               orig_n_recvs);
        return;
    }

    if (s_recvs != orig_s_recvs)
    {
        printf("Size Messages Incorrect! Rank %d got %d, should be %d\n",
               rank,
               s_recvs,
               orig_s_recvs);
        return;
    }

    for (int i = 0; i < n_recvs; i++)
    {
        if (orig_proc_counts[src[i]] != counts[i])
        {
            printf("Rank %d, msgcounts from proc %d incorrect!  Got %d, should be %d\n",
                   rank,
                   src[i],
                   orig_proc_counts[src[i]],
                   counts[i]);
            break;
        }

        for (int j = 0; j < counts[i]; j++)
        {
            if (indices[displs[i] + j] != orig_indices[orig_proc_displs[src[i]] + j])
            {
                printf(
                    "Rank %d, indices from proc %d at pos %d incorrect!  Got %lu, should "
                    "be %lu\n",
                    rank,
                    src[i],
                    j,
                    indices[displs[i] + j],
                    orig_indices[orig_proc_displs[src[i] + j]]);
                break;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;

    int n_iter = 10;
    if (num_procs > 1000)
    {
        n_iter = 100;
    }

    if (argc == 1)
    {
        if (rank == 0)
        {
            printf("Pass Matrix Filename as Command Line Arg!\n");
        }
        MPI_Finalize();
        return 1;
    }
    char* filename = argv[1];

    // Read suitesparse matrix
    ParMat<int> A;
    int not_found = readParMatrix(filename, A);
	if(not_found){return 1;}
    // Form Communication Package (A.send_comm, A.recv_comm)
    form_comm(A);

    MPIL_Comm* xcomm;

    MPIL_Info* xinfo;
    MPIL_Info_init(&xinfo);

    // Form MPIL_Comm initial communicator (should be cheap)
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
        MPIL_Comm_free(&xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("MPIL_Comm_init time %e\n", t0 / n_iter);
    }
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);

    // Split node communicator
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        MPIL_Comm_topo_init(xcomm);
        MPIL_Comm_topo_free(xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("MPIL_Comm_topo_init time %e\n", t0 / n_iter);
    }

    MPIL_Comm_topo_init(xcomm);
    MPIL_Comm_update_locality(xcomm, 4);

    int n_recvs, s_recvs, proc;
    int *src, *rdispls, *recvcounts;
    long* recvvals;

    std::vector<int> proc_count(num_procs, -1);
    std::vector<int> proc_displs(num_procs, -1);
    std::vector<long> orig_indices(A.send_comm.size_msgs + 1);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc              = A.send_comm.procs[i];
        proc_count[proc]  = A.send_comm.counts[i];
        proc_displs[proc] = A.send_comm.ptr[i];
    }
    for (int i = 0; i < A.send_comm.size_msgs; i++)
    {
        orig_indices[i] = A.send_comm.idx[i] + A.first_col;
    }

    // Time Personalized
    MPI_Barrier(MPI_COMM_WORLD);
	MPIL_Set_alltoallv_crs(ALLTOALLV_CRS_PERSONALIZED);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        s_recvs = -1;
        MPIL_Alltoallv_crs(A.recv_comm.n_msgs,
						   A.recv_comm.size_msgs,
						   A.recv_comm.procs.data(),
						   A.recv_comm.counts.data(),
						   A.recv_comm.ptr.data(),
						   MPI_LONG,
						   A.off_proc_columns.data(),
						   &n_recvs,
						   &s_recvs,
						   &src,
						   &recvcounts,
						   &rdispls,
						   MPI_LONG,
						   (void**)&recvvals,
						   xinfo,
						   xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("MPI_Alltoall_crs Time (Personalized VERSION): %e\n", t0 / n_iter);
    }
    compare(n_recvs,
            s_recvs,
            src,
            recvcounts,
            rdispls,
            recvvals,
            A.send_comm.n_msgs,
            A.send_comm.size_msgs,
            proc_count.data(),
            proc_displs.data(),
            orig_indices.data());
    MPIL_Free(src);
    MPIL_Free(recvcounts);
    MPIL_Free(rdispls);
    MPIL_Free(recvvals);

    // Time Nonblocking
    MPI_Barrier(MPI_COMM_WORLD);
	MPIL_Set_alltoallv_crs(ALLTOALLV_CRS_NONBLOCKING);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        s_recvs = -1;
        MPIL_Alltoallv_crs(A.recv_comm.n_msgs,
                                  A.recv_comm.size_msgs,
                                  A.recv_comm.procs.data(),
                                  A.recv_comm.counts.data(),
                                  A.recv_comm.ptr.data(),
                                  MPI_LONG,
                                  A.off_proc_columns.data(),
                                  &n_recvs,
                                  &s_recvs,
                                  &src,
                                  &recvcounts,
                                  &rdispls,
                                  MPI_LONG,
                                  (void**)&recvvals,
                                  xinfo,
                                  xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("MPI_Alltoall_crs Time (Personalized VERSION): %e\n", t0 / n_iter);
    }
    compare(n_recvs,
            s_recvs,
            src,
            recvcounts,
            rdispls,
            recvvals,
            A.send_comm.n_msgs,
            A.send_comm.size_msgs,
            proc_count.data(),
            proc_displs.data(),
            orig_indices.data());
    MPIL_Free(src);
    MPIL_Free(recvcounts);
    MPIL_Free(rdispls);
    MPIL_Free(recvvals);

    // Time Personalized Locality
    MPI_Barrier(MPI_COMM_WORLD);
	MPIL_Set_alltoallv_crs(ALLTOALLV_CRS_PERSONALIZED_LOC);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        s_recvs = -1;
        MPIL_Alltoallv_crs(A.recv_comm.n_msgs,
						   A.recv_comm.size_msgs,
						   A.recv_comm.procs.data(),
						   A.recv_comm.counts.data(),
						   A.recv_comm.ptr.data(),
						   MPI_LONG,
						   A.off_proc_columns.data(),
						   &n_recvs,
						   &s_recvs,
						   &src,
						   &recvcounts,
						   &rdispls,
						   MPI_LONG,
						   (void**)&recvvals,
						   xinfo,
						   xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("MPI_Alltoall_crs Time (Personalized VERSION): %e\n", t0 / n_iter);
    }
    compare(n_recvs,
            s_recvs,
            src,
            recvcounts,
            rdispls,
            recvvals,
            A.send_comm.n_msgs,
            A.send_comm.size_msgs,
            proc_count.data(),
            proc_displs.data(),
            orig_indices.data());
    MPIL_Free(src);
    MPIL_Free(recvcounts);
    MPIL_Free(rdispls);
    MPIL_Free(recvvals);

    // Time Nonblocking Locality
    MPI_Barrier(MPI_COMM_WORLD);
	MPIL_Set_alltoallv_crs(ALLTOALLV_CRS_NONBLOCKING_LOC);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
    {
        s_recvs = -1;
        MPIL_Alltoallv_crs(A.recv_comm.n_msgs,
						  A.recv_comm.size_msgs,
						  A.recv_comm.procs.data(),
						  A.recv_comm.counts.data(),
						  A.recv_comm.ptr.data(),
						  MPI_LONG,
						  A.off_proc_columns.data(),
						  &n_recvs,
						  &s_recvs,
						  &src,
						  &recvcounts,
						  &rdispls,
						  MPI_LONG,
						  (void**)&recvvals,
						  xinfo,
						  xcomm);
    }
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("MPI_Alltoall_crs Time (Personalized VERSION): %e\n", t0 / n_iter);
    }
    compare(n_recvs,
            s_recvs,
            src,
            recvcounts,
            rdispls,
            recvvals,
            A.send_comm.n_msgs,
            A.send_comm.size_msgs,
            proc_count.data(),
            proc_displs.data(),
            orig_indices.data());
    MPIL_Free(src);
    MPIL_Free(recvcounts);
    MPIL_Free(rdispls);
    MPIL_Free(recvvals);

    MPIL_Info_free(&xinfo);
    MPIL_Comm_free(&xcomm);

    MPI_Finalize();
    return 0;
}
