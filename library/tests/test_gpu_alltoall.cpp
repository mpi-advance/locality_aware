#include "../../include/locality_aware.h"
#include "../include/heterogeneous/gpu_alltoall.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void compare_alltoall_results(std::vector<int>& pmpi_alltoall, std::vector<int>& mpix_alltoall, int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int i = 0; i < s*num_procs; i++)
    {
        if (pmpi_alltoall[i] != mpix_alltoall[i])
        {
            fprintf(stderr, "Alltoall ERROR: position %d, pmpi %d, mpix %d\n", 
                    i, pmpi_alltoall[i], mpix_alltoall[i]);
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
    std::vector<int> local_data(max_s*num_procs);
    std::vector<int> pmpi_alltoall(max_s*num_procs);
    std::vector<int> mpix_alltoall(max_s*num_procs);
    std::vector<int> device_data(max_s*num_procs);

    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Comm_device_init(xcomm);
	
    int n_gpus;
    gpuGetDeviceCount(&n_gpus);
    gpuSetDevice(xcomm->rank_gpu);
	std::cout<<"CHECK"<<std::endl;
    int* local_data_d;
    int* alltoall_d;
    gpuMalloc((void**)&local_data_d, 
            max_s*num_procs*sizeof(int));
    gpuMalloc((void**)&alltoall_d, 
            max_s*num_procs*sizeof(int)); 
	std::cout<<"CHECK2"<<std::endl;
    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

		std::cout<<"CHECK 3 "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
		
        // Will only be clean for up to double digit process counts
        for (int i = 0; i < num_procs; i++)
            for (int j = 0; j < s; j++)
                local_data[i*s + j] = rank*10000 + i*100 + j;
		

		std::cout<<"GRID SET "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);	
		
        gpuMemcpy(local_data_d, 
                local_data.data(),
                s*num_procs*sizeof(int),
                gpuMemcpyHostToDevice);

		std::cout<<"after first memCPY "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);	

        // Standard Alltoall
        PMPI_Alltoall(local_data.data(), 
                s,
                MPI_INT, 
                pmpi_alltoall.data(), 
                s, 
                MPI_INT,
                MPI_COMM_WORLD);

		std::cout<<"CHECK 5 "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
        
		// Pairwise Alltoall
        alltoall_pairwise(local_data.data(), 
                s,
                MPI_INT, 
                mpix_alltoall.data(), 
                s, 
                MPI_INT,
                xcomm);
        compare_alltoall_results(pmpi_alltoall, mpix_alltoall, s);

		std::cout<<"pairwise RANK "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);

        // Standard GPU Alltoall
        PMPI_Alltoall(local_data_d,
                s, 
                MPI_INT,
                alltoall_d,
                s,
                MPI_INT,
                MPI_COMM_WORLD);
		
		
	    std::cout<<"before Memcpy "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);	
						
				
				
        gpuMemcpy(device_data.data(), 
                alltoall_d, 
                s*num_procs*sizeof(int), 
                gpuMemcpyDeviceToHost);
				
	    std::cout<<"before comp "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);	
				
				
        compare_alltoall_results(pmpi_alltoall, device_data, s);
        gpuMemset(alltoall_d, 0, s*num_procs*sizeof(int));

		std::cout<<"pmpi RANK "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);

        // GPU-Aware Pairwise Alltoall
        gpu_aware_alltoall_pairwise(
                local_data_d, 
                s, 
                MPI_INT,
                alltoall_d,
                s, 
                MPI_INT,
                xcomm);
        gpuMemcpy(device_data.data(), 
                alltoall_d, 
                s*num_procs*sizeof(int), 
                gpuMemcpyDeviceToHost);
        compare_alltoall_results(pmpi_alltoall, device_data, s);
        gpuMemset(alltoall_d, 0, s*num_procs*sizeof(int));
		
		std::cout<<"pairwise 2 RANK "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);

        // GPU-Aware Nonblocking Alltoall
        gpu_aware_alltoall_nonblocking(
                local_data_d,
                s,
                MPI_INT,
                alltoall_d,
                s,
                MPI_INT,
                xcomm);
        gpuMemcpy(device_data.data(),
                alltoall_d,
                s*num_procs*sizeof(int),
                gpuMemcpyDeviceToHost);
        compare_alltoall_results(pmpi_alltoall, device_data, s);
        gpuMemset(alltoall_d, 0, s*num_procs*sizeof(int));

		std::cout<<"nonblocking RANK "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
		
        // Copy-to-CPU Pairwise Alltoall
        copy_to_cpu_alltoall_pairwise(
                local_data_d, 
                s, 
                MPI_INT,
                alltoall_d,
                s, 
                MPI_INT,
                xcomm);
        gpuMemcpy(device_data.data(), 
                alltoall_d, 
                s*num_procs*sizeof(int), 
                gpuMemcpyDeviceToHost);
        compare_alltoall_results(pmpi_alltoall, device_data, s);
        gpuMemset(alltoall_d, 0, s*num_procs*sizeof(int));

		std::cout<<"cpuRANK "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);

        // Copy-to-CPU Nonblocking Alltoall
        copy_to_cpu_alltoall_nonblocking(
                local_data_d,
                s,
                MPI_INT,
                alltoall_d,
                s,
                MPI_INT,
                xcomm);
        gpuMemcpy(device_data.data(),
                alltoall_d,
                s*num_procs*sizeof(int),
                gpuMemcpyDeviceToHost);
        compare_alltoall_results(pmpi_alltoall, device_data, s);
        gpuMemset(alltoall_d, 0, s*num_procs*sizeof(int));
		
		std::cout<<"cpu non block RANK "<<rank<<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
    }

    gpuFree(local_data_d);
    gpuFree(alltoall_d);

    MPIL_Comm_free(&xcomm);

    MPI_Finalize();
    return 0;
} // end of main() //



