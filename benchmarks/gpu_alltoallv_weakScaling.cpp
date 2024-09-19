

#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>
/**Weak Scaling **/
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

     

    int max_i = 11; // adjust to get larger sizes from here8,9,10,16,20
    int max_s = pow(2, max_i); // Total problem size (constant)
    int s=(max_s*num_procs)/num_procs;
    int n_iter = 100;
    double t0, tfinal;
    


    srand(time(NULL));
    std::vector<double> send_data(s*num_procs);
    std::vector<double> pmpi_alltoallv(s*num_procs);
    std::vector<double> mpix_alltoallv(s*num_procs);

    // Initialize the data to send
    for (int j = 0; j < s; j++) 
        send_data[j] = rand();

    char* cpu_recvbuf; 
    char* cpu_sendbuf;

    double* send_data_d;
    double* recv_data_d;
    cudaMalloc((void**)(&send_data_d), s*num_procs*sizeof(double));
    cudaMalloc((void**)(&recv_data_d), s*num_procs*sizeof(double));
    cudaMemcpy(send_data_d, send_data.data(), s*num_procs*sizeof(double), cudaMemcpyHostToDevice);

     
    MPIX_Comm* locality_comm;
    MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);

    
        std::vector<int> sendcounts(num_procs);
        std::vector<int> sdispls(num_procs);
        std::vector<int> recvcounts(num_procs);
        std::vector<int> rdispls(num_procs);

        // Initialize the sendcounts and recvcounts arrays
        for (int j = 0; j < num_procs; j++) {
            sendcounts[j] = s;
            recvcounts[j] = s;
            sdispls[j] = j * s;
            rdispls[j] = j * s;
           // printf("Process %d, sendcounts[%d]=%d, recvcounts[%d]=%d, sdispls[%d]=%d, rdispls[%d]=%d\n",
             //      rank, j, sendcounts[j], j, recvcounts[j], j, sdispls[j], j, rdispls[j]);
        }
    
     

        int sendcount = 0;
        int recvcount = 0;

        for (int i = 0; i < num_procs; i++)
        {
        sendcount += sendcounts[i];//sendcounts[i];
        recvcount += recvcounts[i];//recvcounts[i];
        }
     
        // allocating memory  cpu_recvbuf, cpu_sendbuf for thecopy_to_cpu_alltoallv_pairwise_extra and copy_to_cpu_alltoallv_nonblocking_extra ;

        cudaMallocHost((void**)&cpu_sendbuf,  sendcount * sizeof(double));
        cudaMallocHost((void**)&cpu_recvbuf,  recvcount * sizeof(double));

    
        MPI_Barrier(MPI_COMM_WORLD);  // Ensuring all processes reach this point
     
        // Standard MPI Implementation
   
        PMPI_Alltoallv(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                       recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);

                       
       
        cudaMemcpy(pmpi_alltoallv.data(), recv_data_d, s*num_procs*sizeof(double), cudaMemcpyDeviceToHost);

      
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        //  GPU-Aware Pairwise Exchange
        gpu_aware_alltoallv_pairwise(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                     recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        cudaMemcpy(mpix_alltoallv.data(), recv_data_d, s*num_procs*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        for (int j = 0; j < s*num_procs; j++)
        {
            if (fabs(pmpi_alltoallv[j] - mpix_alltoallv[j]) > 1e-10)
            {
                fprintf(stderr, "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank, j, pmpi_alltoallv[j], mpix_alltoallv[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // GPU-Aware Nonblocking (P2P)
        gpu_aware_alltoallv_nonblocking(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                        recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        cudaMemcpy(mpix_alltoallv.data(), recv_data_d, s*num_procs*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        for (int j = 0; j < s*num_procs; j++)
        {
            if (fabs(pmpi_alltoallv[j] - mpix_alltoallv[j]) > 1e-10)
            {
                fprintf(stderr, "Rank %d, idx %d, pmpi %e, GA-NB %e\n",
                        rank, j, pmpi_alltoallv[j], mpix_alltoallv[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        gpu_aware_alltoallv_waitany(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                        recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        cudaMemcpy(mpix_alltoallv.data(), recv_data_d, s*num_procs*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        for (int j = 0; j < s*num_procs; j++)
        {
            if (fabs(pmpi_alltoallv[j] - mpix_alltoallv[j]) > 1e-10)
            {
                fprintf(stderr, "Rank %d, idx %d, pmpi %e, GA-NB %e\n",
                        rank, j, pmpi_alltoallv[j], mpix_alltoallv[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        //  Copy to CPU Pairwise Exchange
        copy_to_cpu_alltoallv_pairwise(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                       recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        cudaMemcpy(mpix_alltoallv.data(), recv_data_d, s*num_procs*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        for (int j = 0; j < s*num_procs; j++)
        {
            if (fabs(pmpi_alltoallv[j] - mpix_alltoallv[j]) > 1e-10)
            {
                fprintf(stderr, "Rank %d, idx %d, pmpi %e, C2C-PE %e\n",
                        rank, j, pmpi_alltoallv[j], mpix_alltoallv[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        //  Copy To CPU Nonblocking (P2P)
        copy_to_cpu_alltoallv_nonblocking(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                          recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        cudaMemcpy(mpix_alltoallv.data(), recv_data_d, s*num_procs*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        for (int j = 0; j < s*num_procs; j++)
        {
            if (fabs(pmpi_alltoallv[j] - mpix_alltoallv[j]) > 1e-10)
            {
                fprintf(stderr, "Rank %d, idx %d, pmpi %e, C2C-NB %e\n",
                        rank, j, pmpi_alltoallv[j], mpix_alltoallv[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

   
        copy_to_cpu_alltoallv_pairwise_extra(send_data_d,
                sendcounts.data(),
                sdispls.data(),
                MPI_DOUBLE, 
                recv_data_d,
                recvcounts.data(),
                rdispls.data(),
                MPI_DOUBLE,
                locality_comm,cpu_recvbuf,cpu_sendbuf); 
        cudaMemcpy(mpix_alltoallv.data(), recv_data_d, s*num_procs*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        for (int j = 0; j < s*num_procs; j++)
        {
            if (fabs(pmpi_alltoallv[j] - mpix_alltoallv[j]) > 1e-10)
            {
                fprintf(stderr, "Rank %d, idx %d, pmpi %e, C2C-NB %e\n",
                        rank, j, pmpi_alltoallv[j], mpix_alltoallv[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }


        copy_to_cpu_alltoallv_nonblocking_extra(send_data_d,
                sendcounts.data(),
                sdispls.data(),
                MPI_DOUBLE, 
                recv_data_d,
                recvcounts.data(),
                rdispls.data(),
                MPI_DOUBLE,
                locality_comm,cpu_recvbuf,cpu_sendbuf); 
        cudaMemcpy(mpix_alltoallv.data(), recv_data_d, s*num_procs*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemset(recv_data_d, 0, s*num_procs*sizeof(double));

        for (int j = 0; j < s*num_procs; j++)
        {
            if (fabs(pmpi_alltoallv[j] - mpix_alltoallv[j]) > 1e-10)
            {
                fprintf(stderr, "Rank %d, idx %d, pmpi %e, C2C-NB %e\n",
                        rank, j, pmpi_alltoallv[j], mpix_alltoallv[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        

        // Time PMPI Alltoall
        PMPI_Alltoallv(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                       recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            PMPI_Alltoallv(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                           recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0){
             printf("PMPI_Alltoallv Time %e\n", t0);
             printf("Message Size: %ld bytes\n", s * sizeof(double));
        }
        // Time GPU-Aware Pairwise Exchange
        gpu_aware_alltoallv_pairwise(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                     recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            gpu_aware_alltoallv_pairwise(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                         recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0){
             printf("GPU-Aware Pairwise Exchange Time %e\n", t0);
             printf("Message Size: %ld bytes\n", s * sizeof(double));
        }

        // Time GPU-Aware Nonblocking
        gpu_aware_alltoallv_nonblocking(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                        recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            gpu_aware_alltoallv_nonblocking(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                            recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0){

         printf("GPU-Aware Nonblocking Time %e\n", t0);
         printf("Message Size: %ld bytes\n", s * sizeof(double));
        }

        gpu_aware_alltoallv_waitany(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                        recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
             gpu_aware_alltoallv_waitany(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                            recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0){ 
            printf("GPU-Aware Waitany Time %e\n", t0);
            printf("Message Size: %ld bytes\n", s * sizeof(double));
        }


        // Time Copy-to-CPU Pairwise Exchange
        copy_to_cpu_alltoallv_pairwise(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                       recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            copy_to_cpu_alltoallv_pairwise(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                           recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Copy-to-CPU Pairwise Exchange Time %e\n", t0);
            printf("Message Size: %ld bytes\n", s * sizeof(double));
        }

        // Time Copy-to-CPU Nonblocking
        copy_to_cpu_alltoallv_nonblocking(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                          recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            copy_to_cpu_alltoallv_nonblocking(send_data_d, sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                              recv_data_d, recvcounts.data(), rdispls.data(), MPI_DOUBLE, locality_comm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Copy-to-CPU Nonblocking Time %e\n", t0);
            printf("Message Size: %ld bytes\n", s * sizeof(double));
        }


        copy_to_cpu_alltoallv_pairwise_extra(send_data_d,
                sendcounts.data(),
                sdispls.data(),
                MPI_DOUBLE, 
                recv_data_d,
                recvcounts.data(),
                rdispls.data(),
                MPI_DOUBLE,
                locality_comm,cpu_recvbuf,cpu_sendbuf); 

         cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            copy_to_cpu_alltoallv_pairwise_extra(send_data_d,
                sendcounts.data(),
                sdispls.data(),
                MPI_DOUBLE, 
                recv_data_d,
                recvcounts.data(),
                rdispls.data(),
                MPI_DOUBLE,
                locality_comm,cpu_recvbuf,cpu_sendbuf); 

        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0){
             printf("copy_to_cpu_alltoallv_pairwise_extra Time %e\n", t0);
             printf("Message Size: %ld bytes\n", s * sizeof(double));
        }


        copy_to_cpu_alltoallv_nonblocking_extra(send_data_d,
                sendcounts.data(),
                sdispls.data(),
                MPI_DOUBLE, 
                recv_data_d,
                recvcounts.data(),
                rdispls.data(),
                MPI_DOUBLE,
                locality_comm,cpu_recvbuf,cpu_sendbuf); 

        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            copy_to_cpu_alltoallv_nonblocking_extra(send_data_d,
                sendcounts.data(),
                sdispls.data(),
                MPI_DOUBLE, 
                recv_data_d,
                recvcounts.data(),
                rdispls.data(),
                MPI_DOUBLE,
                locality_comm,cpu_recvbuf,cpu_sendbuf); 

        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("copy_to_cpu_alltoallv_nonblocking_extra Time %e\n", t0);
            printf("Message Size: %ld bytes\n", s * sizeof(double));
        }



   
    //}

    MPIX_Comm_free(locality_comm);
    cudaFree(send_data_d);
    cudaFree(recv_data_d);
    cudaFreeHost(cpu_sendbuf);
    cudaFreeHost(cpu_recvbuf);
  
 
    MPI_Finalize();
    return 0;
}



