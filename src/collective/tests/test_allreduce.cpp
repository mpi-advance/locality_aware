#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void compare_allreduce_results(int pmpi, int mpix)
{
  if (pmpi != mpix)
  {
	fprintf(stderr, "MPIX Allreduce != PMPI, pmpi: %d, mpix: %d\n", pmpi, mpix);
	MPI_Abort(MPI_COMM_WORLD, -1);
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Test Integer Allreduce
  int max_i = 10;
  int max_s = pow(2, max_i);
  srand(time(NULL));
  std::vector<int> local_data(max_s * num_procs);

  MPIX_Comm* locality_comm;
  printf("Initializing locality comm\n");
  MPIX_Comm_init(&locality_comm, MPI_COMM_WORLD);
  printf("Updating locality\n")
  update_locality(locality_comm, 4);

  for (int i =  0; i < max_i; i++)
  {
    int s = pow(2, i);

    for (int j = 0; j < num_procs; j++)
    {
      for (int k = 0; k < s; k++)
	  {
		local_data[j * s + k] = rank * 10000 + j*100 + k;
	  }
    }

	// standard allreduce
	int pmpi_allreduce_sum;
	printf("pmpi\n"):
	PMPI_Allreduce(local_data.data(), &pmpi_allreduce_sum, s, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	
	int mpix_allreduce_sum;
	printf("multileader\n");
	allreduce_multileader(local_data.data(), &mpix_allreduce_sum, s, MPI_INT, MPI_SUM, *locality_comm, 4);
	compare_allreduce_results(pmpi_allreduce_sum, mpix_allreduce_sum);
	
	printf("hierarchical\n");
	allreduce_hierarchical(local_data.data(), &mpix_allreduce_sum, s, MPI_INT, MPI_SUM, *locality_comm);
	compare_allreduce_results(pmpi_allreduce_sum, mpix_allreduce_sum);

	printf("node aware\n");
	allreduce_node_aware(local_data.data(), &mpix_allreduce_sum, s, MPI_INT, MPI_SUM, *locality_comm);
	compare_allreduce_results(pmpi_allreduce_sum, mpix_allreduce_sum);

	printf("locality aware\n");
	allreduce_locality_aware(local_data.data(), &mpix_allreduce_sum, s, MPI_INT, MPI_SUM, *locality_comm, 4);
	compare_allreduce_results(pmpi_allreduce_sum, mpix_allreduce_sum);
  }

  printf("comm free\n");
  MPIX_Comm_free(&locality_comm);
  printf("finalize\n");
  MPI_Finalize();
  return 0;    
}
