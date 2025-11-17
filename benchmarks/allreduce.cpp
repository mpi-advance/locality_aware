#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <cstring>

#include <iostream>
#include <set>
#include <vector>

#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_i  = 10;
    int max_s  = pow(2, max_i);
    int n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));
    std::vector<double> send_data(max_s);
    std::vector<double> pmpi_allreduce(max_s);
    std::vector<double> mpil_allreduce(max_s);
    for (int j = 0; j < max_s; j++)
        send_data[j] = ((float)(rand())) / RAND_MAX;

    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    MPIL_Comm_topo_init(xcomm);

    int local_rank, ppn;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm->local_comm, &ppn);
    MPIL_Comm_leader_init(xcomm, ppn/4);


    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        if (rank == 0)
        {
            printf("Testing Size %d\n", s);
        }

        // Standard MPI Implementation
        PMPI_Allreduce(send_data.data(), pmpi_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Recursive-Doubling
        memset(mpil_allreduce.data(), 0, s*sizeof(int));
        MPIL_Set_allreduce_algorithm(ALLREDUCE_RECURSIVE_DOUBLING);
        MPIL_Allreduce(send_data.data(), mpil_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        for (int j = 0; j < s; j++)
        {
            if (fabs((pmpi_allreduce[j] - mpil_allreduce[j]) / pmpi_allreduce[j]) > 1e-06)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank,
                        j,
                        pmpi_allreduce[j],
                        mpil_allreduce[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // Locality-Aware Dissemination (Node-Aware)
        memset(mpil_allreduce.data(), 0, s*sizeof(int));
        MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_LOC);
        MPIL_Allreduce(send_data.data(), mpil_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        for (int j = 0; j < s; j++)
        {
            if (fabs((pmpi_allreduce[j] - mpil_allreduce[j]) / pmpi_allreduce[j]) > 1e-06)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank,
                        j,
                        pmpi_allreduce[j],
                        mpil_allreduce[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }   
        } 



        // Locality-Aware Dissemination (NUMA-Aware)
        memset(mpil_allreduce.data(), 0, s*sizeof(int));
        MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_ML);
        MPIL_Allreduce(send_data.data(), mpil_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        for (int j = 0; j < s; j++)
        {
            if (fabs((pmpi_allreduce[j] - mpil_allreduce[j]) / pmpi_allreduce[j]) > 1e-06)
            {
                fprintf(stderr,
                        "Rank %d, idx %d, pmpi %e, GA-PE %e\n",
                        rank,
                        j,
                        pmpi_allreduce[j],
                        mpil_allreduce[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }


        // Time PMPI Allreduce
        PMPI_Allreduce(send_data.data(), pmpi_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        PMPI_Allreduce(send_data.data(), pmpi_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tfinal = MPI_Wtime() - t0;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (t0 > 1e-01)
        {
            n_iter = 1;
        }
        else
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int i = 0; i < 10; i++)
            {
                PMPI_Allreduce(send_data.data(), pmpi_allreduce.data(),
                        s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
            tfinal = (MPI_Wtime() - t0)/10;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            
            n_iter = (0.1 / tfinal);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            PMPI_Allreduce(send_data.data(), pmpi_allreduce.data(),
                    s, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0)
        {
            printf("PMPI_Allreduce Time %e\n", t0);
        }



        // Time Recursive Doubling
        MPIL_Set_allreduce_algorithm(ALLREDUCE_RECURSIVE_DOUBLING);
        MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        tfinal = MPI_Wtime() - t0;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (t0 > 1e-01)
        {
            n_iter = 1;
        }
        else
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int i = 0; i < 10; i++)
            {
                MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                        s, MPI_DOUBLE, MPI_SUM, xcomm);
            }
            tfinal = (MPI_Wtime() - t0)/10;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            
            n_iter = (0.1 / tfinal);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                    s, MPI_DOUBLE, MPI_SUM, xcomm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0)
        {
            printf("MPIL Recursive Doubling Allreduce Time %e\n", t0);
        }





        // Time Node-Aware Dissemination
        MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_LOC);
        MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        tfinal = MPI_Wtime() - t0;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (t0 > 1e-01)
        {
            n_iter = 1;
        }
        else
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int i = 0; i < 10; i++)
            {
                MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                        s, MPI_DOUBLE, MPI_SUM, xcomm);
            }
            tfinal = (MPI_Wtime() - t0)/10;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            
            n_iter = (0.1 / tfinal);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                    s, MPI_DOUBLE, MPI_SUM, xcomm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0)
        {
            printf("MPIL Node-Aware Dissemination Allreduce Time %e\n", t0);
        }





        // Time NUMA-Aware Dissemination
        MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_ML);
        MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                s, MPI_DOUBLE, MPI_SUM, xcomm);
        tfinal = MPI_Wtime() - t0;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (t0 > 1e-01)
        {
            n_iter = 1;
        }
        else
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int i = 0; i < 10; i++)
            {
                MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                        s, MPI_DOUBLE, MPI_SUM, xcomm);
            }
            tfinal = (MPI_Wtime() - t0)/10;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            
            n_iter = (0.1 / tfinal);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int k = 0; k < n_iter; k++)
        {
            MPIL_Allreduce(send_data.data(), pmpi_allreduce.data(),
                    s, MPI_DOUBLE, MPI_SUM, xcomm);
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0)
        {
            printf("MPIL NUMA-Aware Dissemination Allreduce Time %e\n", t0);
        }
    }

    MPIL_Comm_free(&xcomm);

    MPI_Finalize();
    return 0;
}
