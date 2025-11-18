#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include <iostream>
#include <set>
#include <vector>

#include "communicator/MPIL_Comm.h"
#include "heterogeneous/gpu_utils.h"
#include "locality_aware.h"

template <typename AllreduceFn, typename... Args>
double time_allreduce(AllreduceFn&& allreduce_fn,
                      Args&&... args)
{
    double t0, tfinal;
    int n_iter;

    // Time PMPI Alltoall
    // 1. Warm Up
    allreduce_fn(std::forward<Args>(args)...);

    // 2. Calculate n_iter (tfinal ~ 1 sec)
    gpuDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    allreduce_fn(std::forward<Args>(args)...);
    tfinal = MPI_Wtime() - t0;
    PMPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (t0 >= 1.0)
    {
        n_iter = 1;
    }
    else
    {
        gpuDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < 10; i++)
            allreduce_fn(std::forward<Args>(args)...);
        tfinal = (MPI_Wtime() - t0) / 10;
        PMPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        n_iter = (1.0 / t0);
    }

    // 3. Measure Timing
    gpuDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; k++)
        allreduce_fn(std::forward<Args>(args)...);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    PMPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return t0;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_i  = 20;
    int max_s  = pow(2, max_i);
    int n_iter = 100;
    srand(time(NULL));
    double t0, tfinal, time;
    std::vector<double> send_data(max_s);
    std::vector<double> pmpi(max_s);
    std::vector<double> mpil(max_s);
    for (int j = 0; j < max_s; j++)
        send_data[j] = ((double)rand()) / RAND_MAX;

    double* send_data_d;
    double* recv_data_d;
    gpuMalloc((void**)(&send_data_d), max_s * sizeof(double));
    gpuMalloc((void**)(&recv_data_d), max_s * sizeof(double));
    gpuMemcpy(send_data_d,
              send_data.data(),
              max_s * sizeof(double),
              gpuMemcpyHostToDevice);

    int num_devices;
    gpuGetDeviceCount(&num_devices);

    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Comm_topo_init(xcomm);
    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    //printf("Rank %d, num devices %d, local rank %d\n", rank, num_devices, local_rank);
    //gpuSetDevice(local_rank);
    gpuSetDevice(0);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0)
        {
            printf("Testing Size %d\n", s);
        }

        // Standard MPI Implementation
        gpuStreamSynchronize(0);
        PMPI_Allreduce(
            send_data_d, recv_data_d, s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        gpuStreamSynchronize(0);
        gpuMemcpy(pmpi.data(),
                  recv_data_d,
                  s * sizeof(double),
                  gpuMemcpyDeviceToHost);
        gpuStreamSynchronize(0);
        gpuMemset(recv_data_d, 0, s * sizeof(double));
        gpuStreamSynchronize(0);


        /*
        // MPI Advance : GPU-Aware Recursive Doubling
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_RECURSIVE_DOUBLING);
        MPIL_Allreduce(send_data_d, recv_data_d, s, MPI_DOUBLE, MPI_SUM, xcomm);
        gpuStreamSynchronize(0);
        gpuMemcpy(mpil.data(),
                  recv_data_d,
                  s * sizeof(double),
                  gpuMemcpyDeviceToHost);
        gpuStreamSynchronize(0);
        gpuMemset(recv_data_d, 0, s * sizeof(double));
        gpuStreamSynchronize(0);
        for (int j = 0; j < s; j++)
        {
            if (fabs((pmpi[j] - mpil[j]) / pmpi[j]) > 1e-06)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank,
                        j,
                        pmpi[j],
                        mpil[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance : GPU-Aware Node-Aware Dissemination
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_LOC);
        MPIL_Allreduce(send_data_d, recv_data_d, s, MPI_DOUBLE, MPI_SUM, xcomm);
        gpuStreamSynchronize(0);
        gpuMemcpy(mpil.data(),
                  recv_data_d,
                  s * sizeof(double),
                  gpuMemcpyDeviceToHost);
        gpuStreamSynchronize(0);
        gpuMemset(recv_data_d, 0, s * sizeof(double));
        gpuStreamSynchronize(0);
        for (int j = 0; j < s; j++)
        {
            if (fabs((pmpi[j] - mpil[j]) / pmpi[j]) > 1e-06)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank,
                        j,
                        pmpi[j],
                        mpil[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        // MPI Advance : GPU-Aware NUMA-Aware Dissemination
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_ML);
        MPIL_Allreduce(send_data_d, recv_data_d, s, MPI_DOUBLE, MPI_SUM, xcomm);
        gpuStreamSynchronize(0);
        gpuMemcpy(mpil.data(),
                  recv_data_d,
                  s * sizeof(double),
                  gpuMemcpyDeviceToHost);
        gpuStreamSynchronize(0);
        gpuMemset(recv_data_d, 0, s * sizeof(double));
        gpuStreamSynchronize(0);
        for (int j = 0; j < s; j++)
        {
            if (fabs((pmpi[j] - mpil[j]) / pmpi[j]) > 1e-06)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank,
                        j,
                        pmpi[j],
                        mpil[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance : CopyToCPU PMPI
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_PMPI);
        gpuStreamSynchronize(0);
        MPIL_Allreduce(send_data_d, recv_data_d, s, MPI_DOUBLE, MPI_SUM, xcomm);
        gpuStreamSynchronize(0);
        gpuMemcpy(mpil.data(),
                  recv_data_d,
                  s * sizeof(double),
                  gpuMemcpyDeviceToHost);
        gpuStreamSynchronize(0);
        gpuMemset(recv_data_d, 0, s * sizeof(double));
        gpuStreamSynchronize(0);
        for (int j = 0; j < s; j++)
        {
            if (fabs((pmpi[j] - mpil[j]) / pmpi[j]) > 1e-06)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank,
                        j,
                        pmpi[j],
                        mpil[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance : CopyToCPU RECURSIVE DOUBLING
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_RECURSIVE_DOUBLING);
        MPIL_Allreduce(send_data_d, recv_data_d, s, MPI_DOUBLE, MPI_SUM, xcomm);
        gpuStreamSynchronize(0);
        gpuMemcpy(mpil.data(),
                  recv_data_d,
                  s * sizeof(double),
                  gpuMemcpyDeviceToHost);
        gpuStreamSynchronize(0);
        gpuMemset(recv_data_d, 0, s * sizeof(double));
        gpuStreamSynchronize(0);
        for (int j = 0; j < s; j++)
        {
            if (fabs((pmpi[j] - mpil[j]) / pmpi[j]) > 1e-06)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank,
                        j,
                        pmpi[j],
                        mpil[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // MPI Advance : CopyToCPU Node-Aware Dissemination
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_DISSEMINATION_LOC);
        MPIL_Allreduce(send_data_d, recv_data_d, s, MPI_DOUBLE, MPI_SUM, xcomm);
        gpuStreamSynchronize(0);
        gpuMemcpy(mpil.data(),
                  recv_data_d,
                  s * sizeof(double),
                  gpuMemcpyDeviceToHost);
        gpuStreamSynchronize(0);
        gpuMemset(recv_data_d, 0, s * sizeof(double));
        gpuStreamSynchronize(0);
        for (int j = 0; j < s; j++)
        {
            if (fabs((pmpi[j] - mpil[j]) / pmpi[j]) > 1e-06)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank,
                        j,
                        pmpi[j],
                        mpil[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
*/

        // MPI Advance : GPU-Aware NUMA-Aware Dissemination
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_DISSEMINATION_ML);
        gpuStreamSynchronize(0);
        MPIL_Allreduce(send_data_d, recv_data_d, s, MPI_DOUBLE, MPI_SUM, xcomm);
        gpuStreamSynchronize(0);
        gpuMemcpy(mpil.data(),
                  recv_data_d,
                  s * sizeof(double),
                  gpuMemcpyDeviceToHost);
        gpuStreamSynchronize(0);
        gpuMemset(recv_data_d, 0, s * sizeof(double));
        gpuStreamSynchronize(0);
        for (int j = 0; j < s; j++)
        {
            if (fabs((pmpi[j] - mpil[j]) / pmpi[j]) > 1e-06)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank,
                        j,
                        pmpi[j],
                        mpil[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }


/*
        // Time PMPI Allreduce
        time = time_allreduce(PMPI_Allreduce, send_data_d, recv_data_d, 
                s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) printf("PMPI Allreduce Time: %e\n", time);

        // Time GPU-Aware Recursive Doubling Allreduce
        //MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_RECURSIVE_DOUBLING);
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_PMPI);
        time = time_allreduce(MPIL_Allreduce, send_data_d, recv_data_d, 
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        if (rank == 0) printf("MPIL GPU-Aware Recursive Doubling Time: %e\n", time);

        // Time GPU-Aware Node-Aware Dissemination Allreduce
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_LOC);
        time = time_allreduce(MPIL_Allreduce, send_data_d, recv_data_d, 
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        if (rank == 0) printf("MPIL GPU-Aware Node-Aware Dissemination Time: %e\n", time);

        // Time GPU-Aware NUMA-Aware Dissemination Allreduce
        MPIL_Set_allreduce_algorithm(ALLREDUCE_GPU_DISSEMINATION_ML);
        time = time_allreduce(MPIL_Allreduce, send_data_d, recv_data_d, 
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        if (rank == 0) printf("MPIL GPU-Aware NUMA-Aware Dissemination Time: %e\n", time);


        // Time CTC PMPI Allreduce
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_PMPI);
        time = time_allreduce(MPIL_Allreduce, send_data_d, recv_data_d, 
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        if (rank == 0) printf("CopyToCPU PMPI Allreduce Time: %e\n", time);

        // Time CopyToCPU Recursive Doubling Allreduce
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_RECURSIVE_DOUBLING);
        time = time_allreduce(MPIL_Allreduce, send_data_d, recv_data_d, 
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        if (rank == 0) printf("MPIL CopyToCPU Recursive Doubling Time: %e\n", time);

        // Time CopyToCPU Node-Aware Dissemination Allreduce
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_DISSEMINATION_LOC);
        time = time_allreduce(MPIL_Allreduce, send_data_d, recv_data_d, 
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        if (rank == 0) printf("MPIL CopyToCPU Node-Aware Dissemination Time: %e\n", time);

        // Time CopyToCPU NUMA-Aware Dissemination Allreduce
        MPIL_Set_allreduce_algorithm(ALLREDUCE_CTC_DISSEMINATION_ML);
        time = time_allreduce(MPIL_Allreduce, send_data_d, recv_data_d, 
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        if (rank == 0) printf("MPIL CopyToCPU NUMA-Aware Dissemination Time: %e\n", time);
*/
    }

    MPIL_Comm_free(&xcomm);

    gpuFree(send_data_d);
    gpuFree(recv_data_d);

    MPI_Finalize();
    return 0;
}
