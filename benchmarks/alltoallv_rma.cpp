#include "mpi_advance.h"
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
//#include "/g/g92/enamug/install/include/caliper/cali.h"
#include "/g/g92/enamug/my_caliper_install/include/caliper/cali.h"
                    
#include "mpi_advance.h"


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    //Weakscaling
  /*  int max_i = 14; // adjust to get larger sizes from here8,9,10,16,20
    int max_s = pow(2, max_i); // Total problem size (constant)
    int s=(max_s*num_procs)/num_procs;
    int n_iter = 100;*/
    //double t0, tfinal;


//strong scaling
    int max_i = 12;
    int max_s = pow(2, max_i);
    int s = (max_s)/ (num_procs);
    int n_iter = 100;

    std::vector<double> send_data(s * num_procs);
    std::vector<double> RMA_winfence_init(s * num_procs);
    std::vector<double> RMA_winlock_init(s * num_procs);
    std::vector<double> recv_data(s * num_procs);
    std::vector<double> validation_recv_data(s * num_procs);

    //  send_data
    for (int i = 0; i < s * num_procs; i++) {
        send_data[i] = rand();
    }
    
    std::vector<int> sendcounts(num_procs, s);
    std::vector<int> recvcounts(num_procs, s);
    std::vector<int> sdispls(num_procs);
    std::vector<int> rdispls(num_procs);

    for (int i = 0; i < num_procs; i++) {
        sdispls[i] = i * s;
        rdispls[i] = i * s;
    }

    
    MPIX_Comm* xcomm;  
  
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    //for the persistent bit
    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    MPIX_Request* xrequest;

    //Timing for PMPI_Alltoallv
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; k++) {
        PMPI_Alltoallv(send_data.data(), sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                       validation_recv_data.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    double pmpi_tfinal = (MPI_Wtime() - t0) / n_iter;

    // reduce timings for PMPI_Alltoallv
    MPI_Reduce(&pmpi_tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("PMPI_Alltoallv Time: %e seconds\n", t0);
        printf("Message Size: %ld bytes\n", s * sizeof(double));
    }


/*
    //Winfence_Init

    MPI_Barrier(xcomm->global_comm);
    double tl = MPI_Wtime();  

    for (int k = 0; k < n_iter; k++) {  
//printf("THIS IS 2");
    alltoallv_rma_winfence_init(send_data.data(),sendcounts.data(),sdispls.data(),MPI_DOUBLE,RMA_winfence_init.data()
        ,recvcounts.data(),rdispls.data(),MPI_DOUBLE, xcomm, xinfo, &xrequest);

   
    MPIX_Request_free(xrequest);
    }
    
    double rma_intwin_final = (MPI_Wtime() - tl) / n_iter;
    MPI_Reduce(&rma_intwin_final, &tl, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
     printf("RMA_winfence_init + Finalise Alltoallv Time (in a for loop): %e seconds\n", tl);
     printf("Message Size: %ld bytes\n", s * sizeof(double));
    }

    //MPI_Barrier(xcomm->global_comm);

    alltoallv_rma_winfence_init(send_data.data(),sendcounts.data(),sdispls.data(),MPI_DOUBLE,RMA_winfence_init.data()
   ,recvcounts.data(),rdispls.data(),MPI_DOUBLE, xcomm, xinfo, &xrequest);

    //MPIX_Request_free(xrequest);
     
     MPI_Barrier(xcomm->global_comm);

    tl = MPI_Wtime();  

    for (int k = 0; k < n_iter; k++) {  
      
        MPIX_Start(xrequest);
        MPIX_Wait(xrequest, MPI_STATUS_IGNORE);

    }
    double rma_fence_final = (MPI_Wtime() - tl) / n_iter;

         //which process takes the longest time 
    MPI_Reduce(&rma_fence_final, &tl, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RMA_winfence_persistent_runtime(start+wait) Alltoallv Time: %e seconds\n", tl);
        printf("Message Size: %ld bytes\n", s * sizeof(double));
    }
    */
    
    //End of Winfence_init

    
    //T winlock_init
    MPI_Barrier(xcomm->global_comm);
    double tl = MPI_Wtime();  
    //This is for accuracy
    
    for (int k = 0; k < n_iter; k++) {  
      
     //printf("rank:%d\n,k =%d\n",rank,k);
     //printf("****a\n");   
     alltoallv_rma_lock_init(send_data.data(),sendcounts.data(),sdispls.data(),MPI_DOUBLE,RMA_winlock_init.data()
     ,recvcounts.data(),rdispls.data(),MPI_DOUBLE, xcomm, xinfo, &xrequest);
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
          printf("Message Size: %ld bytes\n", s * sizeof(double));
      }
     
     
    alltoallv_rma_lock_init(send_data.data(),sendcounts.data(),sdispls.data(),MPI_DOUBLE,RMA_winlock_init.data()
     ,recvcounts.data(),rdispls.data(),MPI_DOUBLE, xcomm, xinfo, &xrequest);

        
     MPI_Barrier(xcomm->global_comm);

     tl = MPI_Wtime();  
     printf("%d: *****Iam at line %d rma lock init*************\n",rank,__LINE__); fflush(stdout);
    for (int k = 0; k < n_iter; k++) 
    {   printf("%d: *****Iam at line %d rma lock init*************k=%d\n",rank,__LINE__,k); fflush(stdout);
        MPIX_Start(xrequest);
        printf("%d: *****Iam at line %d rma lock init*************k=%d\n",rank,__LINE__,k); fflush(stdout);
        MPIX_Wait(xrequest, MPI_STATUS_IGNORE);
        printf("%d: *****Iam at line %d rma lock init*************k=%d\n",rank,__LINE__,k); fflush(stdout);
      
    }
    printf("%d: *****Iam at line %d rma lock init*************\n",rank,__LINE__); fflush(stdout);
    double rma_lock_final = (MPI_Wtime() - tl) / n_iter;
    MPIX_Request_free(xrequest);
 // RMA_lock_init start+ wait
 MPI_Reduce(&rma_lock_final, &tl, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

 if (rank == 0) {

     printf("RMA_second_winlockint Alltoallv(start+wait) Time: %e seconds\n", tl);
     printf("Message Size: %ld bytes\n", s * sizeof(double));
 }

 

   


/*
#if 0
    // Comparing RMA results with PMPI results.
    bool yes11 = true;
    for (int i = 0; i < s * num_procs; i++) {
        if (fabs(RMA_winlock_init[i] - validation_recv_data[i]) > 1e-10) {
            fprintf(stderr, "Validation failed at rank %d, index %d: RMA %f, PMPI %f\n",
                    rank, i, RMA_winlock_init[i], validation_recv_data[i]);
            yes11 = false;
        }
    }

#endif
*/

    MPIX_Comm_free(&xcomm);
    MPI_Finalize();
    return 0;
}
