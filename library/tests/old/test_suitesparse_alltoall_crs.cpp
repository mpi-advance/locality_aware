#include "locality_aware.h"
#include "communicator/MPIL_Comm.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>
#include <set>

#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

void compare_alltoall_crs_results(int n_recvs, int send_msgs, int* recvvals, int* src, 
        std::vector<int>& proc_counts)
{
    if (n_recvs != send_msgs)
    {
        fprintf(stderr, "NRecvs incorrect (%d), should be %d\n", n_recvs, send_msgs);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    for (int i = 0; i < n_recvs; i++)
    {
        if (recvvals[i] != proc_counts[src[i]])
        {
            fprintf(stderr, "RecvVals incorrect: position %d from %d, recvvals %d, should be %d\n", 
                    i, src[i], recvvals[i], proc_counts[src[i]]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void test_matrix(const char* filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    MPIL_Comm* xcomm;
    MPIL_Info* xinfo;

    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Comm_topo_init(xcomm);

    MPIL_Info_init(&xinfo);

    // Update so there are 4 PPN rather than what MPI_Comm_split returns
    update_locality(xcomm, 4);

    // Read suitesparse matrix
    ParMat<int> A;
    readParMatrix(filename, A);
    form_comm(A);
    std::vector<int> proc_counts(num_procs, 0);
    for (int i = 0; i < A.send_comm.n_msgs; i++)
        proc_counts[A.send_comm.procs[i]] = A.send_comm.counts[i];

    int n_recvs;
    int* src;
    int* recvvals;

    /* TEST RMA VERSION */
    n_recvs = -1;
    alltoall_crs_rma(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT, 
            A.recv_comm.counts.data(), &n_recvs, &src, 1, MPI_INT,
            (void**)&recvvals, xinfo, xcomm);
    compare_alltoall_crs_results(n_recvs, A.send_comm.n_msgs, recvvals, src, proc_counts);
    MPIL_Free(src);
    MPIL_Free(recvvals);


    /* TEST PERSONALIZED VERSION */
    n_recvs = -1;
    alltoall_crs_personalized(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
            A.recv_comm.counts.data(), &n_recvs, &src, 1, MPI_INT,
            (void**)&recvvals, xinfo, xcomm);
    compare_alltoall_crs_results(n_recvs, A.send_comm.n_msgs, recvvals, src, proc_counts);
    MPIL_Free(src);
    MPIL_Free(recvvals);


    /* TEST PERSONALIZED LOCALITY VERSION */
    n_recvs = -1;
    alltoall_crs_personalized_loc(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
            A.recv_comm.counts.data(), &n_recvs, &src, 1, MPI_INT,
            (void**)&recvvals, xinfo, xcomm);
    compare_alltoall_crs_results(n_recvs, A.send_comm.n_msgs, recvvals, src, proc_counts);
    MPIL_Free(src);
    MPIL_Free(recvvals);

    /* TEST NONBLOCKING VERSION */
    n_recvs = -1;
    alltoall_crs_nonblocking(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
            A.recv_comm.counts.data(), &n_recvs, &src, 1, MPI_INT,
            (void**)&recvvals, xinfo, xcomm);
    compare_alltoall_crs_results(n_recvs, A.send_comm.n_msgs, recvvals, src, proc_counts);
    MPIL_Free(src);
    MPIL_Free(recvvals);

    /* TEST NONBLOCKING LOCALITY VERSION */
    n_recvs = -1;
    alltoall_crs_nonblocking_loc(A.recv_comm.n_msgs, A.recv_comm.procs.data(), 1, MPI_INT,
            A.recv_comm.counts.data(), &n_recvs, &src, 1, MPI_INT,
            (void**)&recvvals, xinfo, xcomm);
    compare_alltoall_crs_results(n_recvs, A.send_comm.n_msgs, recvvals, src, proc_counts);
    MPIL_Free(src);
    MPIL_Free(recvvals);

    MPIL_Info_free(&xinfo);
    MPIL_Comm_free(&xcomm);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    test_matrix("../../../test_data/dwt_162.pm");
    test_matrix("../../../test_data/odepa400.pm");
    test_matrix("../../../test_data/ww_36_pmec_36.pm");
    test_matrix("../../../test_data/bcsstk01.pm");
    test_matrix("../../../test_data/west0132.pm");
    test_matrix("../../../test_data/gams10a.pm");
    test_matrix("../../../test_data/gams10am.pm");
    test_matrix("../../../test_data/D_10.pm");
    test_matrix("../../../test_data/oscil_dcop_11.pm");
    test_matrix("../../../test_data/tumorAntiAngiogenesis_4.pm");
    test_matrix("../../../test_data/ch5-5-b1.pm");
    test_matrix("../../../test_data/msc01050.pm");
    test_matrix("../../../test_data/SmaGri.pm");
    test_matrix("../../../test_data/radfr1.pm");
    test_matrix("../../../test_data/bibd_49_3.pm");
    test_matrix("../../../test_data/can_1054.pm");
    test_matrix("../../../test_data/can_1072.pm");
    test_matrix("../../../test_data/lp_woodw.pm");
    test_matrix("../../../test_data/lp_sctap2.pm");
    MPI_Finalize();
    return 0;
} // end of main() //



