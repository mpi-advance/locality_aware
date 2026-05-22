#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

#include "locality_aware.h"
#include "gpu_utils.h"

void compare_allgather_results(std::vector<int>& pmpi,
                               std::vector<int>& mpil,
                               int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int i = 0; i < s*num_procs; i++)
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
    std::vector<int> pmpi(max_s*num_procs);
    std::vector<int> mpil(max_s*num_procs);
    std::vector<int> device_data(max_s*num_procs);

    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Comm_device_init(xcomm);

    int ierr;

    int* local_data_d;
    int* allgather_d;
    ierr = gpuMalloc((void**)&local_data_d, 
            max_s * sizeof(int));
    gpu_check(ierr);

    ierr = gpuMalloc((void**)&allgather_d, 
            num_procs * max_s * sizeof(int));
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

        PMPI_Allgather(local_data.data(),
                        s,
                        MPI_INT,
                        pmpi.data(),
                        s,
                        MPI_INT,
                        MPI_COMM_WORLD);

        // Standard Bruck on CPU
        MPIL_Set_allgather_algorithm(ALLGATHER_BRUCK);
        MPIL_Allgather(local_data.data(),
                        s,
                        MPI_INT,
                        mpil.data(),
                        s,
                        MPI_INT,
                        xcomm);
        compare_allgather_results(pmpi, mpil, s);

#if defined(GPU_AWARE)
        // Standard PMPI GPU Allreduce
        PMPI_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        MPI_COMM_WORLD);
        ierr = gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allgather_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // Standard Bruck on GPU
        MPIL_Set_allgather_algorithm(ALLGATHER_GPU_BRUCK);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allgather_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // Standard Ring on GPU
        MPIL_Set_allgather_algorithm(ALLGATHER_GPU_RING);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allgather_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // Standard PMPI on GPU
        MPIL_Set_allgather_algorithm(ALLGATHER_GPU_PMPI);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allgather_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
#endif
        // Standard Bruck Copy-To-CPU
        MPIL_Set_allgather_algorithm(ALLGATHER_CTC_BRUCK);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allgather_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // Standard Ring Copy-To-CPU
        MPIL_Set_allgather_algorithm(ALLGATHER_CTC_RING);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allgather_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // Standard PMPI Copy-To-CPU
        MPIL_Set_allgather_algorithm(ALLGATHER_CTC_PMPI);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        ierr = gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_allgather_results(pmpi, device_data, s);
        ierr = gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
    }

    ierr = gpuFree(local_data_d);
    gpu_check(ierr);
    ierr = gpuFree(allgather_d);
    gpu_check(ierr);

    MPIL_Comm_free(&xcomm);

    MPIL_Finalize();
    MPI_Finalize();
    return 0;
}  // end of main() //
