#include "heterogeneous/gpu_alltoall.h"
#include "heterogeneous/gpu_alltoallv.h"
#include "collective/alltoall.h"
#include "communicator/MPIL_Comm.h"

// ASSUMES 1 CPU CORE PER GPU (Standard for applications)
int gpu_aware_alltoall(alltoall_ftn f,
                       const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm)
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int total_bytes_s = sendcount * send_bytes * num_procs;
    int total_bytes_r = recvcount * recv_bytes * num_procs;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    gpuMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    gpuMallocHost((void**)&cpu_recvbuf, total_bytes_r);

    int ierr = f(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

    gpuFreeHost(cpu_sendbuf);
    gpuFreeHost(cpu_recvbuf);

    return ierr;
}

int gpu_aware_alltoall_nonblocking(const void* sendbuf,
                                   const int sendcount,
                                   MPI_Datatype sendtype,
                                   void* recvbuf,
                                   const int recvcount,
                                   MPI_Datatype recvtype,
                                   MPIL_Comm* comm)
{
    return gpu_aware_alltoall(alltoall_nonblocking,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              comm);
}

int gpu_aware_alltoall_pairwise(const void* sendbuf,
                                const int sendcount,
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcount,
                                MPI_Datatype recvtype,
                                MPIL_Comm* comm)
{
    return gpu_aware_alltoall(alltoall_pairwise,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              comm);
}

int copy_to_cpu_alltoall(alltoall_ftn f,
                         const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm)
{
    int ierr = 0;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int total_bytes_s = sendcount * send_bytes * num_procs;
    int total_bytes_r = recvcount * recv_bytes * num_procs;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    gpuMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    gpuMallocHost((void**)&cpu_recvbuf, total_bytes_r);

    // Copy from GPU to CPU
    ierr += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);

    // Collective Among CPUs
    ierr += f(cpu_sendbuf, sendcount, sendtype, cpu_recvbuf, recvcount, recvtype, comm);

    // Copy from CPU to GPU
    ierr += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);

    gpuFreeHost(cpu_sendbuf);
    gpuFreeHost(cpu_recvbuf);

    return ierr;
}

int copy_to_cpu_alltoall_pairwise(const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm)
{
    return copy_to_cpu_alltoall(alltoall_pairwise,
                                sendbuf,
                                sendcount,
                                sendtype,
                                recvbuf,
                                recvcount,
                                recvtype,
                                comm);
}

int copy_to_cpu_alltoall_nonblocking(const void* sendbuf,
                                     const int sendcount,
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcount,
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm)
{
    return copy_to_cpu_alltoall(alltoall_nonblocking,
                                sendbuf,
                                sendcount,
                                sendtype,
                                recvbuf,
                                recvcount,
                                recvtype,
                                comm);
}

#ifdef OPENMP
int threaded_alltoall_pairwise(const void* sendbuf,
                               const int sendcount,
                               MPI_Datatype sendtype,
                               void* recvbuf,
                               const int recvcount,
                               MPI_Datatype recvtype,
                               MPIL_Comm* comm)
{
    int ierr = 0;

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int total_bytes_s = sendcount * send_bytes * num_procs;
    int total_bytes_r = recvcount * recv_bytes * num_procs;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    gpuMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    gpuMallocHost((void**)&cpu_recvbuf, total_bytes_r);

    // Copy from GPU to CPU
    ierr += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);

    memcpy(cpu_recvbuf + (rank * recvcount * recv_bytes),
           cpu_sendbuf + (rank * sendcount * send_bytes),
           sendcount * send_bytes);

#pragma omp parallel shared(cpu_sendbuf, cpu_recvbuf)
    {
        MPI_Status status;
        int tag;
        MPIL_Comm_tag(comm, &tag);
        int send_proc, recv_proc;
        int send_pos, recv_pos;

        int n_msgs      = num_procs - 1;
        int thread_id   = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        int n_msgs_per_thread = n_msgs / num_threads;
        int extra_msgs        = n_msgs % num_threads;
        int thread_n_msgs     = n_msgs_per_thread;
        if (extra_msgs > thread_id)
        {
            thread_n_msgs++;
        }

        if (thread_n_msgs)
        {
            int idx = thread_id + 1;
            for (int i = 0; i < thread_n_msgs; i++)
            {
                send_proc = rank + idx;
                if (send_proc >= num_procs)
                {
                    send_proc -= num_procs;
                }
                recv_proc = rank - idx;
                if (recv_proc < 0)
                {
                    recv_proc += num_procs;
                }
                send_pos = send_proc * sendcount * send_bytes;
                recv_pos = recv_proc * recvcount * recv_bytes;

                MPI_Sendrecv(cpu_sendbuf + send_pos,
                             sendcount,
                             sendtype,
                             send_proc,
                             tag,
                             cpu_recvbuf + recv_pos,
                             recvcount,
                             recvtype,
                             recv_proc,
                             tag,
                             comm->global_comm,
                             &status);

                idx += num_threads;
            }
        }
    }

    ierr += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);

    gpuFreeHost(cpu_sendbuf);
    gpuFreeHost(cpu_recvbuf);

    return ierr;
}

int threaded_alltoall_nonblocking(const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm)
{
    int num_procs, rank;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int total_bytes_s = sendcount * send_bytes * num_procs;
    int total_bytes_r = recvcount * recv_bytes * num_procs;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    gpuMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    gpuMallocHost((void**)&cpu_recvbuf, total_bytes_r);

    int ierr = 0;
    ierr += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);

    memcpy(cpu_recvbuf + (rank * recvcount * recv_bytes),
           cpu_sendbuf + (rank * sendcount * send_bytes),
           sendcount * send_bytes);

#pragma omp parallel shared(cpu_sendbuf, cpu_recvbuf)
    {
        int tag;
        MPIL_Comm_tag(comm, &tag);
        int send_proc, recv_proc;
        int send_pos, recv_pos;

        int n_msgs      = num_procs - 1;
        int thread_id   = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        int n_msgs_per_thread = n_msgs / num_threads;
        int extra_msgs        = n_msgs % num_threads;
        int thread_n_msgs     = n_msgs_per_thread;
        if (extra_msgs > thread_id)
        {
            thread_n_msgs++;
        }

        if (thread_n_msgs)
        {
            MPI_Request* requests =
                (MPI_Request*)malloc(2 * thread_n_msgs * sizeof(MPI_Request));

            int idx = thread_id + 1;
            for (int i = 0; i < thread_n_msgs; i++)
            {
                send_proc = rank + idx;
                if (send_proc >= num_procs)
                {
                    send_proc -= num_procs;
                }
                recv_proc = rank - idx;
                if (recv_proc < 0)
                {
                    recv_proc += num_procs;
                }
                send_pos = send_proc * sendcount * send_bytes;
                recv_pos = recv_proc * recvcount * recv_bytes;

                MPI_Isend(cpu_sendbuf + send_pos,
                          sendcount,
                          sendtype,
                          send_proc,
                          tag,
                          comm->global_comm,
                          &(requests[i]));
                MPI_Irecv(cpu_recvbuf + recv_pos,
                          recvcount,
                          recvtype,
                          recv_proc,
                          tag,
                          comm->global_comm,
                          &(requests[thread_n_msgs + i]));
                idx += num_threads;
            }

            MPI_Waitall(2 * thread_n_msgs, requests, MPI_STATUSES_IGNORE);

            free(requests);
        }
    }

    ierr += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);
    gpuFreeHost(cpu_sendbuf);
    gpuFreeHost(cpu_recvbuf);

    return ierr;
}

#endif
