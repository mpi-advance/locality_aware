#include "locality_aware.h"

enum AlltoallMethod mpil_alltoall_implementation = ALLTOALL_PAIRWISE;

enum AlltoallvMethod mpil_alltoallv_implementation = ALLTOALLV_PAIRWISE;

enum NeighborAlltoallvMethod mpil_neighbor_alltoallv_implementation =
    NEIGHBOR_ALLTOALLV_STANDARD;

enum NeighborAlltoallvInitMethod mpil_neighbor_alltoallv_init_implementation =
    NEIGHBOR_ALLTOALLV_INIT_STANDARD;

enum AlltoallCRSMethod mpil_alltoall_crs_implementation = ALLTOALL_CRS_PERSONALIZED;

enum AlltoallvCRSMethod mpil_alltoallv_crs_implementation = ALLTOALLV_CRS_PERSONALIZED;

void MPIL_set_alltoall_algorithm(enum AlltoallMethod algorithm)
{
    mpil_alltoall_implementation = (enum AlltoallMethod)algorithm;
}
void MPIL_set_alltoallv_algorithm(enum AlltoallvMethod algorithm)
{
    mpil_alltoallv_implementation = (enum AlltoallvMethod)algorithm;
}

void MPIL_set_alltoallv_neighbor_alogorithm(enum NeighborAlltoallvMethod algorithm)
{
    mpil_neighbor_alltoallv_implementation = (enum NeighborAlltoallvMethod)algorithm;
}

void MPIL_set_alltoallv_neighbor_init_alogorithm(
    enum NeighborAlltoallvInitMethod algorithm)
{
    mpil_neighbor_alltoallv_init_implementation =
        (enum NeighborAlltoallvInitMethod)algorithm;
}

void MPIL_set_alltoall_crs(enum AlltoallCRSMethod algorithm)
{
    mpil_alltoall_crs_implementation = (enum AlltoallCRSMethod)algorithm;
}

void MPIL_set_alltoallv_crs(enum AlltoallvCRSMethod algorithm)
{
    mpil_alltoallv_crs_implementation = (enum AlltoallvCRSMethod)algorithm;
}