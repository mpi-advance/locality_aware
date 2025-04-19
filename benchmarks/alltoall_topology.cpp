#include <vector>

#include <math.h>
#include <mpi.h>
#include "mpi_advance.h"
#include ""

template <void (*T)(void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPIX_Comm*)>
int estimate_iters(const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    int testIterations = 5;
    for (int i = 0; i < testIterations; i++)
    {
        T(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }
    double tFinal = MPI_WTime() - t0 / testIterations;
    MPI_Allreduce(&tFinal, &t0, MPI_DOUBLE, MPI_MAX, 0, comm);

    int n_iters = (5.0 / tFinal) + 1;
    if (tFinal > 1.0)
    {
        n_iters = 1;
    } 

    return n_iters;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int numIterations = 1000;
    int maxSize = pow(2, numIterations);
    // Experiment 1 (system MPI alltoall vs MPI advanced pairwise vs nonblocking)

    int rank;
    int numProcs;

    double t0, tFinal;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORD, &numProcs);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    std::vector<int> localData(maxSize * numProcs);
    std::vector<int> stdAlltoall(maxSize * numProcs);
    std::vector<int> pairwiseAlltoall(maxSize * numProcs);
    std::vector<int> nonblockingAlltoall(maxSize * numProcs);

    int size = 2;
    for (int j = 0; j < numProcs; j++)
    {
        for (int k = 0; k < size; k++)
        {
            localData[j * size + k] = rank * 10000 + j * 100 + k;
        }
    }

    int standardIters = estimate_iters<PMP_Alltoall>(localData.data(), size, MPI_INT, stdAlltoall.data(), size, MPI_INT, xcomm);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < standardIters; i++)
    {
        int size = pow(2, i);
        for (int j = 0; j < numProcs; j++)
        {
            for (int k = 0; k < size; k++)
            {
                localData[j * size + k] = rank * 10000 + j * 100 + k;
            }
        }

        PMPI_Alltoall(local_data.data(),
                      size,
                      MPI_INT,
                      stdAlltoall.data(),
                      size,
                      MPI_INT,
                      xcomm);
    }

    tFinal = MPI_Wtime() - t0;
    MPI_Reduce(&tFinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, xcomm);

    if (rank ==0)
    {
      printf("MPI_Alltoall Time (System Version): %e\n", tFinal / standardIters);
    }

    size = 2;    
    for (int j = 0; j < numProcs; j++)
    {
      for (int k = 0; k < size; k++)
      {
        localData[j * size + k] = rank * 10000 + j * 100 + k;
      }
    }

    int pairwiseIters = estimateIters<alltoall_pairwise>(local_data.data(), 
      size,
      MPI_INT,
      pairwiseAlltoall.data(),
      size,
      MPI_INT,
      xcomm);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < pairwiseIters; i++)
    {
      int size = pow(2, i);
      for (int j = 0; j < numProcs; j++)
      {
        for (int k = 0; k < size; k++)
        {
          localData[j * size + k] = rank * 10000 + j * 100 + k;
        }
      }

      alltoall_pairwise(local_data.data(),
  			size,
	  		MPI_INT,
		  	pairwiseAlltoall.data(),
			  size,
			  MPI_INT,
			  xcomm);
    }     

    tFinal = MPI_Wtime() - t0;
    MPI_Reduce(&tFinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, xcomm);

    if (rank == 0)
    {
      printf("alltoall_pairwise Time: %e\n", tFinal / pairwiseIters);
    }

    size = 2;    
    for (int j = 0; j < numProcs; j++)
    {
      for (int k = 0; k < size; k++)
      {
        localData[j * size + k] = rank * 10000 + j * 100 + k;
      }
    }

    int nonblockingIters = estimateIters<alltoall_nonblocking>(localData.data(),
      size,
      MPI_INT,
      nonblockingData.data(),
      size,
      MPI_INT,
      xcomm);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < nonblockingIters; i++)
    {
      int size = pow(2, i);
      for (int j = 0; j < numProcs; j++)
      {
        for (int k = 0; k < size; k++)
        {
          localData[j * size + k] = rank * 10000 + j * 100 + k;
        }
      }

      alltoall_nonblocking(localData.data(),
        size,
        MPI_INT,
        nonblockingData.data(),
        size,
        MPI_INT,
        xcomm);
    }

    tFinal = MPI_Wtime() - t0;
    MPI_Reduce(&tFinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, xcomm);
    
    if (rank == 0)
    {
      printf("alltoall_nonblocking Time: %e\n", tFinal / nonblockingIters);
    }
}
