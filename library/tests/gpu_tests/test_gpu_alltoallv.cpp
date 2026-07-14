#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

#include "gpu_utils.h"
#include "locality_aware.h"

void compare_alltoall_results(std::vector<int>& pmpi_alltoall,
                              std::vector<int>& mpix_alltoall,
                              int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int i = 0; i < s * num_procs; i++)
    {
        if (pmpi_alltoall[i] != mpix_alltoall[i])
        {
            fprintf(stderr,
                    "Alltoallv ERROR: position %d, pmpi %d, mpix %d\n",
                    i,
                    pmpi_alltoall[i],
                    mpix_alltoall[i]);
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
    std::vector<int> local_data(max_s * num_procs);
    std::vector<int> pmpi_alltoall(max_s * num_procs);
    std::vector<int> mpix_alltoall(max_s * num_procs);
    std::vector<int> device_data(max_s * num_procs);
    std::vector<int> sendcounts(num_procs);
    std::vector<int> recvcounts(num_procs);
    std::vector<int> sdispls(num_procs+1);
    std::vector<int> rdispls(num_procs+1);

    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Comm_device_init(xcomm);

    int ierr;

    int* local_data_d;
    int* alltoall_d;
    ierr = gpuMalloc((void**)&local_data_d, max_s * num_procs * sizeof(int));
    gpu_check(ierr);
    ierr = gpuMalloc((void**)&alltoall_d, max_s * num_procs * sizeof(int));
    gpu_check(ierr);

    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        for (int i = 0; i < num_procs; i++)
        {
            for (int j = 0; j < s; j++)
            {
                local_data[i * s + j] = rank * 10000 + i * 100 + j;
            }
            sendcounts[i] = s;
            recvcounts[i] = s;
            sdispls[i+1] = sdispls[i] + s;
            rdispls[i+1] = rdispls[i] + s;
        }
        ierr = gpuMemcpyAsync(local_data_d,
                  local_data.data(),
                  s * num_procs * sizeof(int),
                  gpuMemcpyHostToDevice,
                  0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);

        // Standard Alltoall
        PMPI_Alltoallv(local_data.data(),
                      sendcounts.data(),
                      sdispls.data(),
                      MPI_INT,
                      pmpi_alltoall.data(),
                      recvcounts.data(),
                      rdispls.data(),
                      MPI_INT,
                      MPI_COMM_WORLD);

        // Pairwise Alltoall
        MPIL_Set_alltoallv_algorithm(ALLTOALLV_PAIRWISE);
        MPIL_Alltoallv(local_data.data(),
                      sendcounts.data(),
                      sdispls.data(),
                      MPI_INT,
                      mpix_alltoall.data(),
                      recvcounts.data(),
                      rdispls.data(),
                      MPI_INT,
                      xcomm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);
        if (rank == 0) printf("MPIL and PMPI equivalent on CPUs\n");

#if defined(GPU_AWARE)
        // Standard GPU Alltoall
        PMPI_Alltoallv(local_data_d,
                sendcounts.data(),
                sdispls.data(),
                MPI_INT,
                alltoall_d,
                recvcounts.data(),
                rdispls.data(),
                MPI_INT,
                MPI_COMM_WORLD);
        ierr = gpuMemcpyAsync(device_data.data(),
                  alltoall_d,
                  s * num_procs * sizeof(int),
                  gpuMemcpyDeviceToHost,
                  0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_alltoall_results(pmpi_alltoall, device_data, s);
        ierr = gpuMemsetAsync(alltoall_d, 0, s * num_procs * sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        if (rank == 0) printf("PMPI equivalent on CPU and GPU\n");

        // GPU-Aware Pairwise Alltoall
        MPIL_Set_alltoallv_algorithm(ALLTOALLV_GPU_PAIRWISE);
        MPIL_Alltoallv(local_data_d,
                    sendcounts.data(),
                    sdispls.data(),
                    MPI_INT,
                    alltoall_d,
                    recvcounts.data(),
                    rdispls.data(),
                    MPI_INT,
                    xcomm);
        ierr = gpuMemcpyAsync(device_data.data(),
                  alltoall_d,
                  s * num_procs * sizeof(int),
                  gpuMemcpyDeviceToHost,
                  0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_alltoall_results(pmpi_alltoall, device_data, s);
        ierr = gpuMemsetAsync(alltoall_d, 0, s * num_procs * sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        if (rank == 0) printf("GPU Pairwise equivalent to PMPI\n");

        // GPU-Aware Nonblocking Alltoall
        MPIL_Set_alltoallv_algorithm(ALLTOALLV_GPU_NONBLOCKING);
        MPIL_Alltoallv(local_data_d,
                    sendcounts.data(),
                    sdispls.data(),
                    MPI_INT,
                    alltoall_d,
                    recvcounts.data(),
                    rdispls.data(),
                    MPI_INT,
                    xcomm);
        ierr = gpuMemcpyAsync(device_data.data(),
                  alltoall_d,
                  s * num_procs * sizeof(int),
                  gpuMemcpyDeviceToHost,
                  0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_alltoall_results(pmpi_alltoall, device_data, s);
        ierr = gpuMemsetAsync(alltoall_d, 0, s * num_procs * sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        if (rank == 0) printf("GPU Nonblocking equivalent to PMPI\n");
#endif

        // Copy-to-CPU Pairwise Alltoall
        MPIL_Set_alltoallv_algorithm(ALLTOALLV_CTC_PAIRWISE);
        MPIL_Alltoallv(local_data_d,
                    sendcounts.data(),
                    sdispls.data(),
                    MPI_INT,
                    alltoall_d,
                    recvcounts.data(),
                    rdispls.data(),
                    MPI_INT,
                    xcomm);
        ierr = gpuMemcpyAsync(device_data.data(),
                  alltoall_d,
                  s * num_procs * sizeof(int),
                  gpuMemcpyDeviceToHost,
                  0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_alltoall_results(pmpi_alltoall, device_data, s);
        ierr = gpuMemsetAsync(alltoall_d, 0, s * num_procs * sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        if (rank == 0) printf("C2C pairwise equivalent to PMPI\n");

        // Copy-to-CPU Nonblocking Alltoall
        MPIL_Set_alltoallv_algorithm(ALLTOALLV_CTC_NONBLOCKING);
        MPIL_Alltoallv(local_data_d,
                    sendcounts.data(),
                    sdispls.data(),
                    MPI_INT,
                    alltoall_d,
                    recvcounts.data(),
                    rdispls.data(),
                    MPI_INT,
                    xcomm);
        ierr = gpuMemcpyAsync(device_data.data(),
                  alltoall_d,
                  s * num_procs * sizeof(int),
                  gpuMemcpyDeviceToHost,
                  0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        compare_alltoall_results(pmpi_alltoall, device_data, s);
        ierr = gpuMemsetAsync(alltoall_d, 0, s * num_procs * sizeof(int), 0);
        gpu_check(ierr);
        ierr = gpuStreamSynchronize(0);
        gpu_check(ierr);
        if (rank == 0) printf("C2C nonblocking equivalent to PMPI\n");
    }

    ierr = gpuFree(local_data_d);
    gpu_check(ierr);
    ierr = gpuFree(alltoall_d);
    gpu_check(ierr);

    MPIL_Comm_free(&xcomm);

    MPIL_Finalize();
    MPI_Finalize();
    return 0;
}  // end of main() //
