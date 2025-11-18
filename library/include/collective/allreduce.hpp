#ifndef MPI_ADVANCE_ALLREDUCE_HPP
#define MPI_ADVANCE_ALLREDUCE_HPP

#include <mpi.h>
#include <stdlib.h>
#include "allreduce.h"
#include "communicator/MPIL_Comm.h"

template <typename AllocFn, typename FreeFn>
int allreduce_impl(allreduce_helper_ftn f,
                   const void* sendbuf,
                   void* recvbuf,
                   int count,
                   MPI_Datatype datatype,
                   MPI_Op op,
                   MPIL_Comm* comm,
                   AllocFn alloc_fn,
                   FreeFn free_fn)
{
    if (count == 0)
        return MPI_SUCCESS;

    int type_size;
    MPI_Type_size(datatype, &type_size);

    void* tmpbuf = nullptr;
    alloc_fn(&tmpbuf, type_size * count);

    f(sendbuf, tmpbuf, recvbuf, count, datatype, op, comm);

    free_fn(tmpbuf);
    return MPI_SUCCESS;
}

#endif
