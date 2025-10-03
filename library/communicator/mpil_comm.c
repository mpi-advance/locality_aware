#include "../../include/communicator/mpil_comm.h"

int MPIL_Comm_init(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm)
{
    int rank, num_procs;
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &num_procs);

    MPIL_Comm* xcomm   = (MPIL_Comm*)malloc(sizeof(MPIL_Comm));
    xcomm->global_comm = global_comm;

    xcomm->local_comm = MPI_COMM_NULL;
    xcomm->group_comm = MPI_COMM_NULL;

    xcomm->leader_comm       = MPI_COMM_NULL;
    xcomm->leader_group_comm = MPI_COMM_NULL;
    xcomm->leader_local_comm = MPI_COMM_NULL;

    xcomm->neighbor_comm = MPI_COMM_NULL;

    xcomm->win       = MPI_WIN_NULL;
    xcomm->win_array = NULL;
    xcomm->win_bytes = 0;

    xcomm->requests   = NULL;
    xcomm->statuses   = NULL;
    xcomm->n_requests = 0;

    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &(xcomm->max_tag), &flag);
    xcomm->tag = 126 % xcomm->max_tag;

    xcomm->global_rank_to_local = NULL;
    xcomm->global_rank_to_node  = NULL;
    xcomm->ordered_global_ranks = NULL;

#ifdef GPU
    xcomm->gpus_per_node = 0;
#endif

    *xcomm_ptr = xcomm;

    return MPI_SUCCESS;
}

int MPIL_Comm_topo_init(MPIL_Comm* xcomm)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    // Split global comm into local (per node) communicators
    MPI_Comm_split_type(xcomm->global_comm,
                        MPI_COMM_TYPE_SHARED,
                        rank,
                        MPI_INFO_NULL,
                        &(xcomm->local_comm));

    int local_rank, ppn;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm->local_comm, &ppn);

    // Split global comm into group (per local rank) communicators
    MPI_Comm_split(xcomm->global_comm, local_rank, rank, &(xcomm->group_comm));

    int node;
    MPI_Comm_rank(xcomm->group_comm, &node);

    // Gather arrays for get_node, get_local, and get_global methods
    // These arrays allow for these methods to work with any ordering
    // No longer relying on SMP ordering of processes to nodes!
    // Does rely on constant ppn
    xcomm->global_rank_to_local = (int*)malloc(num_procs * sizeof(int));
    xcomm->global_rank_to_node  = (int*)malloc(num_procs * sizeof(int));
    MPI_Allgather(&local_rank,
                  1,
                  MPI_INT,
                  xcomm->global_rank_to_local,
                  1,
                  MPI_INT,
                  xcomm->global_comm);
    MPI_Allgather(
        &node, 1, MPI_INT, xcomm->global_rank_to_node, 1, MPI_INT, xcomm->global_comm);

    xcomm->ordered_global_ranks = (int*)malloc(num_procs * sizeof(int));
    for (int i = 0; i < num_procs; i++)
    {
        int local                                       = xcomm->global_rank_to_local[i];
        int node                                        = xcomm->global_rank_to_node[i];
        xcomm->ordered_global_ranks[node * ppn + local] = i;
    }

    // Set xcomm variables
    MPI_Comm_size(xcomm->local_comm, &(xcomm->ppn));
    xcomm->num_nodes = ((num_procs - 1) / xcomm->ppn) + 1;
    xcomm->rank_node = get_node(xcomm, rank);

    return MPI_SUCCESS;
}

int MPIL_Comm_leader_init(MPIL_Comm* xcomm, int procs_per_leader)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    MPI_Comm_split(
        xcomm->global_comm, rank / procs_per_leader, rank, &(xcomm->leader_comm));

    int leader_rank;
    MPI_Comm_rank(xcomm->leader_comm, &leader_rank);

    MPI_Comm_split(xcomm->global_comm, leader_rank, rank, &(xcomm->leader_group_comm));

    if (xcomm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(xcomm);
    }

    MPI_Comm_split(xcomm->local_comm, leader_rank, rank, &(xcomm->leader_local_comm));

    return MPI_SUCCESS;
}

int MPIL_Comm_device_init(MPIL_Comm* xcomm)
{
#ifdef GPU
    if (xcomm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(xcomm);
    }

    int local_rank, ierr;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    ierr = gpuGetDeviceCount(&(xcomm->gpus_per_node));
    gpu_check(ierr);
    if (xcomm->gpus_per_node)
    {
        xcomm->rank_gpu = local_rank;
        ierr            = gpuStreamCreate(&(xcomm->proc_stream));
        gpu_check(ierr);
    }
#endif

    return MPI_SUCCESS;
}

int MPIL_Comm_win_init(MPIL_Comm* xcomm, int bytes, int type_bytes)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    xcomm->win_bytes      = bytes;
    xcomm->win_type_bytes = type_bytes;
    MPI_Alloc_mem(xcomm->win_bytes, MPI_INFO_NULL, &(xcomm->win_array));
    MPI_Win_create(xcomm->win_array,
                   xcomm->win_bytes,
                   xcomm->win_type_bytes,
                   MPI_INFO_NULL,
                   xcomm->global_comm,
                   &(xcomm->win));

    return MPI_SUCCESS;
}

int MPIL_Comm_req_resize(MPIL_Comm* xcomm, int n)
{
    if (n <= 0)
    {
        return MPI_SUCCESS;
    }

    xcomm->n_requests = n;
    xcomm->requests   = (MPI_Request*)realloc(xcomm->requests, n * sizeof(MPI_Request));
    xcomm->statuses   = (MPI_Status*)realloc(xcomm->statuses, n * sizeof(MPI_Status));

    return MPI_SUCCESS;
}

int MPIL_Comm_tag(MPIL_Comm* xcomm, int* tag)
{
    *tag       = xcomm->tag;
    xcomm->tag = ((xcomm->tag + 1) % xcomm->max_tag);

    return MPI_SUCCESS;
}

int MPIL_Comm_free(MPIL_Comm** xcomm_ptr)
{
    MPIL_Comm* xcomm = *xcomm_ptr;

    if (xcomm->n_requests > 0)
    {
        free(xcomm->requests);
    }

    if (xcomm->neighbor_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->neighbor_comm));
    }

    MPIL_Comm_topo_free(xcomm);
    MPIL_Comm_leader_free(xcomm);
    MPIL_Comm_win_free(xcomm);
    MPIL_Comm_device_free(xcomm);

    free(xcomm);

    return MPI_SUCCESS;
}

int MPIL_Comm_topo_free(MPIL_Comm* xcomm)
{
    if (xcomm->local_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->local_comm));
    }
    if (xcomm->group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->group_comm));
    }

    if (xcomm->global_rank_to_local != NULL)
    {
        free(xcomm->global_rank_to_local);
    }
    if (xcomm->global_rank_to_node != NULL)
    {
        free(xcomm->global_rank_to_node);
    }
    if (xcomm->ordered_global_ranks != NULL)
    {
        free(xcomm->ordered_global_ranks);
    }

    return MPI_SUCCESS;
}

int MPIL_Comm_leader_free(MPIL_Comm* xcomm)
{
    if (xcomm->leader_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->leader_comm));
    }
    if (xcomm->leader_group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->leader_group_comm));
    }
    if (xcomm->leader_local_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->leader_local_comm));
    }

    return MPI_SUCCESS;
}

int MPIL_Comm_win_free(MPIL_Comm* xcomm)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    if (xcomm->win != MPI_WIN_NULL)
    {
        MPI_Win_free(&(xcomm->win));
    }
    if (xcomm->win_array != NULL)
    {
        MPI_Free_mem(xcomm->win_array);
    }
    xcomm->win_bytes      = 0;
    xcomm->win_type_bytes = 0;

    return MPI_SUCCESS;
}

int MPIL_Comm_device_free(MPIL_Comm* xcomm)
{
#ifdef GPU
    int ierr = gpuSuccess;
    if (xcomm->gpus_per_node)
    {
        ierr = gpuStreamDestroy(xcomm->proc_stream);
    }
    gpu_check(ierr);
#endif

    return MPI_SUCCESS;
}

/****  Topology Functions   ****/
int get_node(const MPIL_Comm* data, const int proc)
{
    return data->global_rank_to_node[proc];
}

int get_local_proc(const MPIL_Comm* data, const int proc)
{
    return data->global_rank_to_local[proc];
}

int get_global_proc(const MPIL_Comm* data, const int node, const int local_proc)
{
    return data->ordered_global_ranks[local_proc + (node * data->ppn)];
}

// For testing purposes
// Manually update aggregation size (ppn)
void update_locality(MPIL_Comm* xcomm, int ppn)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    if (xcomm->local_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->local_comm));
    }
    if (xcomm->group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->group_comm));
    }

    MPI_Comm_split(xcomm->global_comm, rank / ppn, rank, &(xcomm->local_comm));

    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_split(xcomm->global_comm, local_rank, rank, &(xcomm->group_comm));

    int node;
    MPI_Comm_rank(xcomm->group_comm, &node);

    if (xcomm->global_rank_to_local == NULL)
    {
        xcomm->global_rank_to_local = (int*)malloc(num_procs * sizeof(int));
    }

    if (xcomm->global_rank_to_node == NULL)
    {
        xcomm->global_rank_to_node = (int*)malloc(num_procs * sizeof(int));
    }

    MPI_Allgather(&local_rank,
                  1,
                  MPI_INT,
                  xcomm->global_rank_to_local,
                  1,
                  MPI_INT,
                  xcomm->global_comm);
    MPI_Allgather(
        &node, 1, MPI_INT, xcomm->global_rank_to_node, 1, MPI_INT, xcomm->global_comm);

    if (xcomm->ordered_global_ranks == NULL)
    {
        xcomm->ordered_global_ranks = (int*)malloc(num_procs * sizeof(int));
    }

    for (int i = 0; i < num_procs; i++)
    {
        int local                                       = xcomm->global_rank_to_local[i];
        int node                                        = xcomm->global_rank_to_node[i];
        xcomm->ordered_global_ranks[node * ppn + local] = i;
    }

    MPI_Comm_size(xcomm->local_comm, &(xcomm->ppn));
    xcomm->num_nodes = ((num_procs - 1) / xcomm->ppn) + 1;
    xcomm->rank_node = get_node(xcomm, rank);
}
