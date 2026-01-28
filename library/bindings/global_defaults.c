#include "locality_aware.h"

// Default algorithms
enum AlltoallMethod mpil_alltoall_implementation          = ALLTOALL_PAIRWISE;
enum AlltoallvMethod mpil_alltoallv_implementation        = ALLTOALLV_PAIRWISE;
enum AlltoallCRSMethod mpil_alltoall_crs_implementation   = ALLTOALL_CRS_PERSONALIZED;
enum AlltoallvCRSMethod mpil_alltoallv_crs_implementation = ALLTOALLV_CRS_PERSONALIZED;
enum NeighborAlltoallvMethod mpil_neighbor_alltoallv_implementation =
    NEIGHBOR_ALLTOALLV_STANDARD;
enum NeighborAlltoallvInitMethod mpil_neighbor_alltoallv_init_implementation =
    NEIGHBOR_ALLTOALLV_INIT_STANDARD;

int MPIL_Set_alltoall_algorithm(enum AlltoallMethod algorithm)
{
    mpil_alltoall_implementation = (enum AlltoallMethod)algorithm;
    return MPI_SUCCESS;
}
int MPIL_Set_alltoallv_algorithm(enum AlltoallvMethod algorithm)
{
    mpil_alltoallv_implementation = (enum AlltoallvMethod)algorithm;
    return MPI_SUCCESS;
}

int MPIL_Set_alltoallv_neighbor_alogorithm(enum NeighborAlltoallvMethod algorithm)
{
    mpil_neighbor_alltoallv_implementation = (enum NeighborAlltoallvMethod)algorithm;
    return MPI_SUCCESS;
}

int MPIL_Set_alltoallv_neighbor_init_alogorithm(
    enum NeighborAlltoallvInitMethod algorithm)
{
    mpil_neighbor_alltoallv_init_implementation =
        (enum NeighborAlltoallvInitMethod)algorithm;
    return MPI_SUCCESS;
}

int MPIL_Set_alltoall_crs(enum AlltoallCRSMethod algorithm)
{
    mpil_alltoall_crs_implementation = (enum AlltoallCRSMethod)algorithm;
    return MPI_SUCCESS;
}




int MPIL_Set_split(enum split_implementation algorithm)
{
    split_implementation = (enum Split)algorithm;
    return MPI_SUCCESS;
}



int Set_split(enum AlltoallvCRSMethod algorithm)
{
    mpil_alltoallv_crs_implementation = (enum AlltoallvCRSMethod)algorithm;
    return MPI_SUCCESS;
}