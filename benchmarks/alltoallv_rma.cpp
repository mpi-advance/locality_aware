#include "mpi_advance.h"
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <algorithm>

//#include "/g/g92/enamug/install/include/caliper/cali.h"
#include "/g/g92/enamug/my_caliper_install/include/caliper/cali.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Strong scaling
    int max_i = 12;
    int max_s = pow(2, max_i);
    int s = max_s / num_procs;
    int n_iter = 100;

    srand(rank);
    std::vector<int> sendcounts(num_procs);
    std::vector<int> recvcounts(num_procs);
    std::vector<int> sdispls(num_procs);
    std::vector<int> rdispls(num_procs);

    for (int i = 0; i < num_procs; i++) {
        double factor = 0.9 + 0.2 * ((double)rand() / RAND_MAX); // [0.9, 1.1]
        sendcounts[i] = static_cast<int>(s * factor);//Because we want an int
       // recvcounts[i] = sendcounts[i];//
    }

    
    // Hey I will be sending these  sendcounts so every body knows what it's receiving
MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    //Modified send and recive displacements as well
    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int i = 1; i < num_procs; i++) 
    {
        sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
    }

    int total_send = 0;
    int total_recv = 0;

    
    for(int i=0;i<num_procs;i++)
    {
        total_send += sendcounts[i];
        total_recv += recvcounts[i];
    }
  
    
    std::vector<double> send_data(total_send);
    std::vector<double> RMA_winfence_init(total_recv);
    std::vector<double> RMA_winlock_init(total_recv);
    std::vector<double> recv_data(total_recv);
    std::vector<double> validation_recv_data(total_recv);


 
  for (int i = 0; i < num_procs; i++) {
    for (int j = 0; j < sendcounts[i]; j++) {
        send_data[sdispls[i] + j] = (double)rank;
    }
}


for (int j = 0; j < num_procs; j++) {
    int bytes_sent = sendcounts[j] * sizeof(double);
    int bytes_expected = recvcounts[j] * sizeof(double);
    printf("Rank %d sends %d bytes to Rank %d, but Rank %d expects %d bytes from Rank %d\n",
           rank, bytes_sent, j, j, bytes_expected, rank);
}
fflush(stdout);


    
    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    MPIX_Request* xrequest;

    // PMPI_Alltoallv timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; k++) {
        PMPI_Alltoallv(send_data.data(), sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                       validation_recv_data.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    double pmpi_tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&pmpi_tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    //range in bytes
    int min_send_count_bytes = *std::min_element(sendcounts.begin(), sendcounts.end()) * (int)sizeof(double);
    int max_send_count_bytes = *std::max_element(sendcounts.begin(), sendcounts.end()) * (int)sizeof(double);

    if (rank == 0) {
        printf("PMPI_Alltoallv Time: %e seconds\n", t0);
        printf("Message Size Range: [%d, %d] bytes\n", min_send_count_bytes, max_send_count_bytes);
    }//printing msg size

    MPI_Barrier(xcomm->global_comm);
    double tl = MPI_Wtime();  

    for (int k = 0; k < n_iter; k++) {  
    // RMA_winfence persistent
    alltoallv_rma_winfence_init(send_data.data(), sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                RMA_winfence_init.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                                xcomm, xinfo, &xrequest);
    MPI_Barrier(xcomm->global_comm);

    MPIX_Request_free(xrequest);
    }

   
    double rma_fence_final = (MPI_Wtime() - tl) / n_iter;

    MPI_Reduce(&rma_fence_final, &tl, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("looped RMA_winfence_init+Finalize Time: %e seconds\n", tl);
        printf("Message Size Range: [%d, %d] bytes\n", min_send_count_bytes, max_send_count_bytes);
    }


    alltoallv_rma_winfence_init(send_data.data(),sendcounts.data(),sdispls.data(),MPI_DOUBLE,RMA_winfence_init.data()
   ,recvcounts.data(),rdispls.data(),MPI_DOUBLE, xcomm, xinfo, &xrequest);

    //MPIX_Request_free(xrequest);
     
     MPI_Barrier(xcomm->global_comm);

    tl = MPI_Wtime();  

    for (int k = 0; k < n_iter; k++) {  
         //printf("%d: *****Iam at line %d ************k=%d\n",rank,__LINE__,k); fflush(stdout); 
           	    
        MPIX_Start(xrequest);
	  //printf("%d: *****Iam at line %d *************k=%d\n",rank,__LINE__,k); fflush(stdout);
        MPIX_Wait(xrequest, MPI_STATUS_IGNORE);

	  //printf("%d: *****Iam at line %d *************k=%d\n",rank,__LINE__,k); fflush(stdout);

    }
     rma_fence_final = (MPI_Wtime() - tl) / n_iter;
    
    MPIX_Request_free(xrequest);
         //which process takes the longest time 
    MPI_Reduce(&rma_fence_final, &tl, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RMA_winfence_persistent_runtime(start+wait) Alltoallv Time: %e seconds\n", tl);
        printf("Message Size Range: [%d, %d] bytes\n", min_send_count_bytes, max_send_count_bytes);
       // printf("Message Size: %ld bytes\n", s * sizeof(double));
    }


    int lock_errors = 0;
    for (int src = 0; src < num_procs; src++) {
        for (int j = 0; j < recvcounts[src]; j++) {
            int index = rdispls[src] + j;
            if (RMA_winfence_init[index] != (double)src) {
                printf("Rank %d mismatch from src %d at elem %d: got %.1f, expected %.1f\n",
                       rank, src, j, RMA_winfence_init[index], (double)src);
                lock_errors++;
            }
        }
    }
    if (lock_errors == 0 && rank == 0) {
        printf("RMA_winfence_init correctness: PASSED\n");
    }
  
      

    // RMA_winlock persistent
   // printf("*************hey b4 entered rma_lock_start\n");
   // fflush(stdout);
    MPI_Barrier(xcomm->global_comm);
     tl = MPI_Wtime();  
    //This is for accuracy
    for (int k = 0; k < n_iter; k++) {  
     //printf("rank:%d\n,k =%d\n",rank,k);
     //printf("****a\n"); 
    alltoallv_rma_lock_init(send_data.data(), sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                            RMA_winlock_init.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                            xcomm, xinfo, &xrequest);
                           // printf("***********after init\n");
                            fflush(stdout);

    // printf("****b\n");   
    MPIX_Request_free(xrequest);
    MPIX_Comm_win_free(xcomm);
    //printf("****c\n");   
    }  
    double rma_intlock_final = (MPI_Wtime() - tl) / n_iter;

    // RMA Alltoallv Time
    MPI_Reduce(&rma_intlock_final, &tl, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RMA_winlock_init Alltoallv+ Finalize Time (in a for loop): %e seconds\n", tl);
        printf("Message Size Range: [%d, %d] bytes\n", min_send_count_bytes, max_send_count_bytes);
        //printf("Message Size: %ld bytes\n", s * sizeof(double));
    }

    alltoallv_rma_lock_init(send_data.data(),sendcounts.data(),sdispls.data(),MPI_DOUBLE,RMA_winlock_init.data()
     ,recvcounts.data(),rdispls.data(),MPI_DOUBLE, xcomm, xinfo, &xrequest);

    MPI_Barrier(xcomm->global_comm);
    t0 = MPI_Wtime();
    //printf("%d: *****Iam at line %d rma lock init*************\n",rank,__LINE__); fflush(stdout);
   // printf("***********1111\n");
    for (int k = 0; k < n_iter; k++) {
          printf("%d: *****Iam at line %d rma lock init*************k=%d\n",rank,__LINE__,k); fflush(stdout);
        MPIX_Start(xrequest);
        printf("Rank %d: starting MPIX_Start at iter %d\n", rank, k); fflush(stdout);
        MPIX_Wait(xrequest, MPI_STATUS_IGNORE);
        printf("Rank %d: completed iteration %d\n", rank, k); fflush(stdout);
    }
    double rma_lock_final = (MPI_Wtime() - t0) / n_iter;
    MPIX_Request_free(xrequest);
    //printf("***********4444\n");

 

    MPI_Reduce(&rma_lock_final, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RMA_winlock_persistent Alltoallv Time: %e seconds\n", t0);
        printf("Message Size Range: [%d, %d] bytes\n", min_send_count_bytes, max_send_count_bytes);
    }

    int errors = 0;
    for (int src = 0; src < num_procs; src++) {
        for (int j = 0; j < recvcounts[src]; j++) {
            int index = rdispls[src] + j;
            if (RMA_winlock_init[index] != (double)src) {
                printf("Rank %d mismatch from src %d at elem %d: got %.1f, expected %.1f\n",
                       rank, src, j, RMA_winlock_init[index], (double)src);
                errors++;
            }
        }
    }
    if (errors == 0 && rank == 0) {
        printf("RMA_winlock_init correctness: PASSED\n");
    }
  


    MPIX_Comm_free(&xcomm);
    MPI_Finalize();
    return 0;
}
