#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

#include "locality_aware.h"
#include "gpu_utils.h"

void compare_allreduce_results(std::vector<int>& pmpi,
                               std::vector<int>& mpil,
                               int s)
{
    for (int i = 0; i < s; i++)
    {
        if (pmpi[i] != mpil[i])
        {
            fprintf(stderr,
                    "ERROR: position %d, pmpi %d, mpix %d\n",
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
    MPIL_Init(MPI_COMM_WORLD);

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

    int ierr;

    int* local_data_d;
    int* allreduce_d;
    ierr = gpuMalloc((void**)&local_data_d, max_s * sizeof(int));
    gpu_check(ierr);
    ierr = gpuMalloc((void**)&allreduce_d, max_s * sizeof(int));
    gpu_check(ierr);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int i = 0; i < s; i++)
        {
            local_data[i] = rank * 10000 + i;
        }
        ierr = gpuMemcpyAsync(local_data_d,
                  local_data.data(),
                  s * sizeof(int),
                  gpuMemcpyHostToDevice,
                  0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

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
        PMPI_Allreduce(local_data_d, allreduce_d, s, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        ierr = gpuMemcpyAsync(device_data.data(), allreduce_d, 
                s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allreduce_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allreduce_d, 0, s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // Standard Recursive Doubling on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_RECURSIVE_DOUBLING);
        MPIL_Allreduce(local_data_d, allreduce_d, s, MPI_INT, MPI_SUM, xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allreduce_d, 
                s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allreduce_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allreduce_d, 0, s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        
        // Node-Aware Dissemination on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_LOC);
        MPIL_Allreduce(local_data_d, allreduce_d, s, MPI_INT, MPI_SUM, xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allreduce_d, 
                s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allreduce_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allreduce_d, 0, s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        
        // NUMA-Aware Dissemination on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_ML);
        MPIL_Allreduce(local_data_d, allreduce_d, s, MPI_INT, MPI_SUM, xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allreduce_d, 
                s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allreduce_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allreduce_d, 0, s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // HIGH-Radix Dissemination on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_RADIX);
        MPIL_Allreduce(local_data_d, allreduce_d, s, MPI_INT, MPI_SUM, xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allreduce_d, 
                s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allreduce_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allreduce_d, 0, s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

#endif
        // CopyToCPU Standard Recursive Doubling on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_RECURSIVE_DOUBLING);
        MPIL_Allreduce(local_data_d, allreduce_d, s, MPI_INT, MPI_SUM, xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allreduce_d, 
                s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allreduce_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allreduce_d, 0, s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // CopyToCPU Node-Aware Dissemination on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_DISSEMINATION_LOC);
        MPIL_Allreduce(local_data_d, allreduce_d, s, MPI_INT, MPI_SUM, xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allreduce_d, 
                s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allreduce_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allreduce_d, 0, s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // CopyToCPU NUMA-Aware Dissemination on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_DISSEMINATION_ML);
        MPIL_Allreduce(local_data_d, allreduce_d, s, MPI_INT, MPI_SUM, xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allreduce_d, 
                s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allreduce_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allreduce_d, 0, s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // CopyToCPU HIGH-Radix Dissemination on GPU
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_DISSEMINATION_RADIX);
        MPIL_Allreduce(local_data_d, allreduce_d, s, MPI_INT, MPI_SUM, xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allreduce_d, 
                s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allreduce_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allreduce_d, 0, s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
    }

    ierr = gpuFree(local_data_d);
    gpu_check(ierr);
    ierr = gpuFree(allreduce_d);
    gpu_check(ierr);

    MPIL_Comm_free(&xcomm);

    MPIL_Finalize();
    MPI_Finalize();
    return 0;
}  // end of main() //
