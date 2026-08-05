#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"
#include "neighborhood/neighbor.h"
#include "string.h"

// Standard, non-persistent neighbor collective
int neighbor_alltoallv_coll_a2a(const void* sendbuf,
                                const int sendcounts[],
                                const int sdispls[],
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                MPI_Datatype recvtype,
                                MPIL_Topo* topo,
                                MPIL_Comm* comm)
{
    int sbytes, rbytes;
    MPI_Type_size(sendtype, &sbytes);
    MPI_Type_size(recvtype, &rbytes);

    int proc;
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    const char* char_sendbuf = (char*)sendbuf;
    char* char_recvbuf = (char*)recvbuf;

    std::vector<int> coll_sendcounts(num_procs, 0);
    std::vector<int> coll_sdispls(num_procs+1);
    std::vector<int> coll_recvcounts(num_procs, 0);
    std::vector<int> coll_rdispls(num_procs+1);
    
    int ssize = 0;
    for (int i = 0; i < topo->outdegree; i++)
    {
        coll_sendcounts[topo->destinations[i]] = sendcounts[i];
    }
    coll_sdispls[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        coll_sdispls[i+1] = coll_sdispls[i] + coll_sendcounts[i];
    }
    std::vector<char> coll_sendbuf(coll_sdispls[num_procs]*sbytes);

    for (int i = 0; i < topo->outdegree; i++)
    {
        proc = topo->destinations[i];
        MPI_Sendrecv(&(char_sendbuf[sdispls[i]*sbytes]),
                coll_sendcounts[proc],
                sendtype,
                0, 
                0,
                &(coll_sendbuf[coll_sdispls[proc]*sbytes]),
                coll_sendcounts[proc],
                sendtype, 
                0,
                0,
                MPI_COMM_SELF, 
                MPI_STATUS_IGNORE);
    }



    int rsize = 0;
    for (int i = 0; i < topo->indegree; i++)
    {
        coll_recvcounts[topo->sources[i]] = recvcounts[i];
    }
    coll_rdispls[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        coll_rdispls[i+1] = coll_rdispls[i] + coll_recvcounts[i];
    }
    std::vector<char> coll_recvbuf(coll_rdispls[num_procs]*rbytes);


    MPIL_Alltoallv(coll_sendbuf.data(), coll_sendcounts.data(), coll_sdispls.data(), sendtype,
            coll_recvbuf.data(), coll_recvcounts.data(), coll_rdispls.data(), recvtype, 
            comm);

    for (int i = 0; i < topo->indegree; i++)
    {
        proc = topo->sources[i];
        MPI_Sendrecv(&(coll_recvbuf[coll_rdispls[proc]*rbytes]),
                coll_recvcounts[proc],
                recvtype,
                0, 
                0,
                &(char_recvbuf[rdispls[i]*rbytes]),
                coll_recvcounts[proc],
                recvtype, 
                0,
                0,
                MPI_COMM_SELF, 
                MPI_STATUS_IGNORE);


    }


 
    return MPI_SUCCESS;
}

