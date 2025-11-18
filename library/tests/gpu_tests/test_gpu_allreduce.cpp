#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

#include "heterogeneous/gpu_utils.h"
#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

void compare_allreduce_results(std::vector<int>& pmpi,
                               std::vector<int>& mpil,
                               int s)
{
    for (int i = 0; i < s; i++)
    {
        if (pmpi[i] != mpil[i])
        {
            fprintf(stderr,
                    "Alltoall ERROR: position %d, pmpi %d, mpix %d\n",
                    i,
                    pmpi[i],
                    mpil[i]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Test Integer Alltoall
    int max_i = 10;
    int max_s = pow(2, max_i);
    srand(time(NULL));
    std::vector<int> local_data(max_s);
    std::vector<int> pmpi(max_s);
    std::vector<int> mpil(max_s);
    std::vector<int> device_data(max_s);

    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Comm_device_init(xcomm);

    int n_gpus;
    gpuGetDeviceCount(&n_gpus);
    gpuSetDevice(xcomm->rank_gpu);

    int* local_data_d;
    int* alltoall_d;
    gpuMalloc((void**)&local_data_d, max_s * sizeof(int));
    gpuMalloc((void**)&alltoall_d, max_s * sizeof(int));

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int i = 0; i < s; i++)
        {
            local_data[i] = rank * 10000 + i;
        }
        gpuMemcpy(local_data_d,
                  local_data.data(),
                  s * sizeof(int),
                  gpuMemcpyHostToDevice);

        PMPI_Allreduce(local_data.data(),
                       pmpi.data(),
                       s,
                       MPI_INT,
                       MPI_SUM,
                       MPI_COMM_WORLD);

        // Standard Recursive Doubling on CPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_RECURSIVE_DOUBLING);
        MPIL_Allreduce(
                local_data.data(), mpil.data(), s, MPI_INT, MPI_SUM, xcomm);
        compare_allreduce_results(pmpi, mpil, s);

#if defined(GPU_AWARE)
        // Standard PMPI GPU Allreduce
        PMPI_Allreduce(local_data_d, alltoall_d, s, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        gpuMemcpy(device_data.data(), alltoall_d, s*sizeof(int), gpuMemcpyDeviceToHost);
        compare_allreduce_results(pmpi, device_data, s);
        gpuMemset(alltoall_d, 0, s*sizeof(int));

        // Standard Recursive Doubling on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_RECURSIVE_DOUBLING);
        MPIL_Allreduce(local_data_d, alltoall_d, s, MPI_INT, MPI_SUM, xcomm);
        gpuMemcpy(device_data.data(), alltoall_d, s*sizeof(int), gpuMemcpyDeviceToHost);
        compare_allreduce_results(pmpi, device_data, s);
        gpuMemset(alltoall_d, 0, s*sizeof(int));
        
        // Node-Aware Dissemination on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_LOC);
        MPIL_Allreduce(local_data_d, alltoall_d, s, MPI_INT, MPI_SUM, xcomm);
        gpuMemcpy(device_data.data(), alltoall_d, s*sizeof(int), gpuMemcpyDeviceToHost);
        compare_allreduce_results(pmpi, device_data, s);
        gpuMemset(alltoall_d, 0, s*sizeof(int));
        
        // NUMA-Aware Dissemination on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_ML);
        MPIL_Allreduce(local_data_d, alltoall_d, s, MPI_INT, MPI_SUM, xcomm);
        gpuMemcpy(device_data.data(), alltoall_d, s*sizeof(int), gpuMemcpyDeviceToHost);
        compare_allreduce_results(pmpi, device_data, s);
        gpuMemset(alltoall_d, 0, s*sizeof(int));
#endif

        // CopyToCPU Standard Recursive Doubling on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_RECURSIVE_DOUBLING);
        MPIL_Allreduce(local_data_d, alltoall_d, s, MPI_INT, MPI_SUM, xcomm);
        gpuMemcpy(device_data.data(), alltoall_d, s*sizeof(int), gpuMemcpyDeviceToHost);
        compare_allreduce_results(pmpi, device_data, s);
        gpuMemset(alltoall_d, 0, s*sizeof(int));

        // CopyToCPU Node-Aware Dissemination on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_LOC);
        MPIL_Allreduce(local_data_d, alltoall_d, s, MPI_INT, MPI_SUM, xcomm);
        gpuMemcpy(device_data.data(), alltoall_d, s*sizeof(int), gpuMemcpyDeviceToHost);
        compare_allreduce_results(pmpi, device_data, s);
        gpuMemset(alltoall_d, 0, s*sizeof(int));

        // CopyToCPU NUMA-Aware Dissemination on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_ML);
        MPIL_Allreduce(local_data_d, alltoall_d, s, MPI_INT, MPI_SUM, xcomm);
        gpuMemcpy(device_data.data(), alltoall_d, s*sizeof(int), gpuMemcpyDeviceToHost);
        compare_allreduce_results(pmpi, device_data, s);
        gpuMemset(alltoall_d, 0, s*sizeof(int));
    }

    gpuFree(local_data_d);
    gpuFree(alltoall_d);

    MPIL_Comm_free(&xcomm);

    MPI_Finalize();
    return 0;
}  // end of main() //
