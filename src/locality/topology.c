#include "topology.h"

int MPIX_Comm_init(MPIX_Comm** xcomm_ptr, MPI_Comm global_comm)
{
    int rank, num_procs;
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &num_procs);

    MPIX_Comm* xcomm = (MPIX_Comm*)malloc(sizeof(MPIX_Comm));
    xcomm->global_comm = global_comm;

    xcomm->local_comm = MPI_COMM_NULL;
    xcomm->group_comm = MPI_COMM_NULL;

    xcomm->neighbor_comm = MPI_COMM_NULL;

    xcomm->win = MPI_WIN_NULL;
    xcomm->win_array = NULL;
    xcomm->win_bytes = 0;

    xcomm->requests = NULL;
    xcomm->n_requests = 0;

    xcomm->gpus_per_node = 0;

    *xcomm_ptr = xcomm;

    return MPI_SUCCESS;
}


int MPIX_Comm_topo_init(MPIX_Comm* xcomm)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    MPI_Comm_split_type(xcomm->global_comm,
        MPI_COMM_TYPE_SHARED,
        rank,
        MPI_INFO_NULL,
        &(xcomm->local_comm));

    MPI_Comm_size(xcomm->local_comm, &(xcomm->ppn));
    xcomm->num_nodes = ((num_procs-1) / xcomm->ppn) + 1;
    xcomm->rank_node = get_node(xcomm, rank);

    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);

    MPI_Comm_split(xcomm->global_comm,
            local_rank,
            rank,
            &(xcomm->group_comm));

    return MPI_SUCCESS;
}

int MPIX_Comm_device_init(MPIX_Comm* xcomm)
{
#ifdef GPU
    if (xcomm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(xcomm);

    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    gpuGetDeviceCount(&(xcomm->gpus_per_node));
    if (xcomm->gpus_per_node)
    {
        xcomm->rank_gpu = local_rank;
        gpuStreamCreate(&(xcomm->proc_stream));
    }
#endif

    return MPI_SUCCESS;
}

int MPIX_Comm_win_init(MPIX_Comm* xcomm, int bytes, int type_bytes)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);


    xcomm->win_bytes = bytes;
    xcomm->win_type_bytes = type_bytes;
    MPI_Alloc_mem(xcomm->win_bytes, MPI_INFO_NULL, &(xcomm->win_array));
    MPI_Win_create(xcomm->win_array, xcomm->win_bytes, 
            xcomm->win_type_bytes, MPI_INFO_NULL, 
            xcomm->global_comm, &(xcomm->win));

    return MPI_SUCCESS;
}

int MPIX_Comm_req_resize(MPIX_Comm* xcomm, int n)
{
    if (n <= 0) return MPI_SUCCESS;

    xcomm->n_requests = n;
    xcomm->requests = (MPI_Request*)realloc(xcomm->requests, n*sizeof(MPI_Request));

    return MPI_SUCCESS;
}

int MPIX_Comm_free(MPIX_Comm** xcomm_ptr)
{
    MPIX_Comm* xcomm = *xcomm_ptr;

    if (xcomm->n_requests > 0)
        free(xcomm->requests);

    if (xcomm->neighbor_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(xcomm->neighbor_comm));

    MPIX_Comm_topo_free(xcomm);
    MPIX_Comm_win_free(xcomm);
    MPIX_Comm_device_free(xcomm);

    free(xcomm);

    return MPI_SUCCESS;
}

int MPIX_Comm_topo_free(MPIX_Comm* xcomm)
{
   if (xcomm->local_comm != MPI_COMM_NULL)
      MPI_Comm_free(&(xcomm->local_comm));
   if (xcomm->group_comm != MPI_COMM_NULL)
       MPI_Comm_free(&(xcomm->group_comm));

    return MPI_SUCCESS;
}

int MPIX_Comm_win_free(MPIX_Comm* xcomm)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

   if (xcomm->win != MPI_WIN_NULL)
       MPI_Win_free(&(xcomm->win));
   if (xcomm->win_array != NULL)
       MPI_Free_mem(xcomm->win_array);
   xcomm->win_bytes = 0;
   xcomm->win_type_bytes = 0;

    return MPI_SUCCESS;
}

int MPIX_Comm_device_free(MPIX_Comm* xcomm)
{
#ifdef GPU
    if (xcomm->gpus_per_node)
        gpuStreamDestroy(xcomm->proc_stream);
#endif

    return MPI_SUCCESS;
}




/****  Topology Functions   ****/
int get_node(const MPIX_Comm* data, const int proc)
{
    return proc / data->ppn;
}

int get_local_proc(const MPIX_Comm* data, const int proc)
{
    return proc % data->ppn;
}

int get_global_proc(const MPIX_Comm* data, const int node, const int local_proc)
{
    return local_proc + (node * data->ppn);
}

// For testing purposes
// Manually update aggregation size (ppn)
void update_locality(MPIX_Comm* xcomm, int ppn)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    if (xcomm->local_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(xcomm->local_comm));
    if (xcomm->group_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(xcomm->group_comm));

    MPI_Comm_split(xcomm->global_comm,
        rank / ppn,
        rank,
        &(xcomm->local_comm));

    MPI_Comm_size(xcomm->local_comm, &(xcomm->ppn));
    xcomm->num_nodes = ((num_procs - 1) / xcomm->ppn) + 1;
    xcomm->rank_node = get_node(xcomm, rank);

    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_split(xcomm->global_comm,
        local_rank,
        rank,
        &(xcomm->group_comm));
}

