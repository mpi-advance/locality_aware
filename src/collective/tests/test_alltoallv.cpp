#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void compare_alltoallv_results(std::vector<int>& pmpi, std::vector<int>& mpix, int s)
{
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int j = 0; j < s*num_procs; j++)
    {
        if (pmpi[j] != mpix[j])
        {
            fprintf(stderr, "MPIL Alltoallv != PMPI, position %d, pmpi %d, mpix %d\n", 
                    j, pmpi[j], mpix[j]);
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

    std::vector<int> pmpi_alltoallv(max_s*num_procs);
    std::vector<int> mpix_alltoallv(max_s*num_procs);

    std::vector<int> sizes(num_procs);
    std::vector<int> displs(num_procs+1);

    MPIL_Comm* xcomm;
    MPIL_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 4);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);

        // Will only be clean for up to double digit process counts
        displs[0] = 0;
        for (int j = 0; j < num_procs; j++)
        {
            for (int k = 0; k < s; k++)
                local_data[j*s + k] = rank*10000 + j*100 + k;
            sizes[j] = s;
            displs[j+1] = displs[j] + s;
        }


        PMPI_Alltoallv(local_data.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT, 
                pmpi_alltoallv.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT,
                MPI_COMM_WORLD);

        std::fill(mpix_alltoallv.begin(), mpix_alltoallv.end(), 0);
        MPIL_Alltoallv(local_data.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT, 
                mpix_alltoallv.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT,
                xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpix_alltoallv, s);

        std::fill(mpix_alltoallv.begin(), mpix_alltoallv.end(), 0);
        alltoallv_pairwise(local_data.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT, 
                mpix_alltoallv.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT,
                xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpix_alltoallv, s);

        std::fill(mpix_alltoallv.begin(), mpix_alltoallv.end(), 0);
        alltoallv_nonblocking(local_data.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT, 
                mpix_alltoallv.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT,
                xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpix_alltoallv, s);

        std::fill(mpix_alltoallv.begin(), mpix_alltoallv.end(), 0);
        alltoallv_batch(local_data.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT, 
                mpix_alltoallv.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT,
                xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpix_alltoallv, s);

        std::fill(mpix_alltoallv.begin(), mpix_alltoallv.end(), 0);
        alltoallv_batch_async(local_data.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT, 
                mpix_alltoallv.data(), 
                sizes.data(),
                displs.data(),
                MPI_INT,
                xcomm);
        compare_alltoallv_results(pmpi_alltoallv, mpix_alltoallv, s);

    }

    MPIL_Comm_free(&xcomm);


    MPI_Finalize();
    return 0;
} // end of main() //



