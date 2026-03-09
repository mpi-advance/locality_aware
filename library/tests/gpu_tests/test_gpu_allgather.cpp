#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

#include "locality_aware.h"

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

    int num_devices;
    gpuGetDeviceCount(&num_devices);
    int local_rank;
    MPIL_Comm_local_rank(xcomm, &local_rank);
    if (local_rank < num_devices)
        gpuSetDevice(local_rank);
    else // assuming only single device visible
        gpuSetDevice(0);

    int* local_data_d;
    int* allgather_d;
    gpuMalloc((void**)&local_data_d, 
            max_s * sizeof(int));
    gpuMalloc((void**)&allgather_d, 
            num_procs * max_s * sizeof(int));

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int i = 0; i < s; i++)
        {
            local_data[i] = rank * 10000 + i;
        }
        gpuMemcpyAsync(local_data_d,
                  local_data.data(),
                  s * sizeof(int),
                  gpuMemcpyHostToDevice,
                  0);
        gpuStreamSynchronize(0);

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
        gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpuStreamSynchronize(0);
        compare_allgather_results(pmpi, device_data, s);
        gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpuStreamSynchronize(0);

        // Standard Bruck on GPU
        MPIL_Set_allgather_algorithm(ALLGATHER_GPU_BRUCK);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpuStreamSynchronize(0);
        compare_allgather_results(pmpi, device_data, s);
        gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpuStreamSynchronize(0);

        // Standard Ring on GPU
        MPIL_Set_allgather_algorithm(ALLGATHER_GPU_RING);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpuStreamSynchronize(0);
        compare_allgather_results(pmpi, device_data, s);
        gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpuStreamSynchronize(0);

        // Standard PMPI on GPU
        MPIL_Set_allgather_algorithm(ALLGATHER_GPU_PMPI);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpuStreamSynchronize(0);
        compare_allgather_results(pmpi, device_data, s);
        gpuMemsetAsync(allgather_d, 0, num_procs*s*sizeof(int), 0);
        gpuStreamSynchronize(0);
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
        gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpuStreamSynchronize(0);
        compare_allgather_results(pmpi, device_data, s);
        gpuMemset(allgather_d, 0, num_procs*s*sizeof(int));

        // Standard Ring Copy-To-CPU
        MPIL_Set_allgather_algorithm(ALLGATHER_CTC_RING);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpuStreamSynchronize(0);
        compare_allgather_results(pmpi, device_data, s);
        gpuMemset(allgather_d, 0, num_procs*s*sizeof(int));

        // Standard PMPI Copy-To-CPU
        MPIL_Set_allgather_algorithm(ALLGATHER_CTC_PMPI);
        MPIL_Allgather(local_data_d,
                        s,
                        MPI_INT,
                        allgather_d,
                        s,
                        MPI_INT,
                        xcomm);
        gpuMemcpyAsync(device_data.data(), allgather_d, 
                num_procs*s*sizeof(int), gpuMemcpyDeviceToHost, 0);
        gpuStreamSynchronize(0);
        compare_allgather_results(pmpi, device_data, s);
        gpuMemset(allgather_d, 0, num_procs*s*sizeof(int));
    }

    gpuFree(local_data_d);
    gpuFree(allgather_d);

    MPIL_Comm_free(&xcomm);

    MPI_Finalize();
    return 0;
}  // end of main() //
