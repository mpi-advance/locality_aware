#include "../../include/collective/alltoall.h"
#include "../../include/collective/alltoallv.h"
#include "../../include/neighborhood/neighbor.h"
#include "../../include/neighborhood/neighbor_init.h"

enum AlltoallMethod mpil_alltoall_implementation = ALLTOALL_PAIRWISE;
enum AlltoallvMethod mpil_alltoallv_implementation = ALLTOALLV_PAIRWISE;
enum NeighborAlltoallvInitMethod mpix_neighbor_alltoallv_init_implementation =
    NEIGHBOR_ALLTOALLV_INIT_STANDARD;
enum NeighborAlltoallvMethod mpix_neighbor_alltoallv_implementation =
    NEIGHBOR_ALLTOALLV_STANDARD;
