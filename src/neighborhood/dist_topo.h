typedef struct _MPIX_Topo
{
    int indegree;
    int* sources;
    int outdegree;
    int* destinations;

} MPIX_Topo;

int MPIX_Topo_dist_graph_adjacent(MPI_Comm comm, 
        int indegree,
        const int sources[],
        int outdegree,
        const int destinations[],
        MPI_Info info,
        MPIX_Topo** mpix_topo_ptr);

int MPIX_Topo_free(MPIX_Topo* topo);
