#include "mpix_comm.h"
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stdlib.h>

int MPIX_Comm_init(MPIX_Comm** xcomm_ptr, MPI_Comm global_comm)
{
    int rank, num_procs;
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &num_procs);

    MPIX_Comm* xcomm = (MPIX_Comm*)malloc(sizeof(MPIX_Comm));
    xcomm->global_comm = global_comm;

    xcomm->local_comm = MPI_COMM_NULL;
    xcomm->group_comm = MPI_COMM_NULL;

    xcomm->leader_comm = MPI_COMM_NULL;
    xcomm->leader_group_comm = MPI_COMM_NULL;
    xcomm->leader_local_comm = MPI_COMM_NULL;

    xcomm->neighbor_comm = MPI_COMM_NULL;

    xcomm->win = MPI_WIN_NULL;
    xcomm->win_array = NULL;
    xcomm->win_bytes = 0;

    xcomm->requests = NULL;
    xcomm->statuses = NULL;
    xcomm->n_requests = 0;

    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &(xcomm->max_tag), &flag);
    xcomm->tag = 126 % xcomm->max_tag;

    xcomm->global_rank_to_local = NULL;
    xcomm->global_rank_to_node = NULL;
    xcomm->ordered_global_ranks = NULL;

#ifdef GPU
    xcomm->gpus_per_node = 0;
#endif

    *xcomm_ptr = xcomm;

    return MPI_SUCCESS;
}


int MPIX_Comm_topo_init(MPIX_Comm* xcomm)
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
    MPI_Comm_split(xcomm->global_comm,
            local_rank,
            rank,
            &(xcomm->group_comm));

    int node;
    MPI_Comm_rank(xcomm->group_comm, &node);

    // Gather arrays for get_node, get_local, and get_global methods
    // These arrays allow for these methods to work with any ordering
    // No longer relying on SMP ordering of processes to nodes!
    // Does rely on constant ppn
    xcomm->global_rank_to_local = (int*)malloc(num_procs*sizeof(int));
    xcomm->global_rank_to_node = (int*)malloc(num_procs*sizeof(int));
    MPI_Allgather(&local_rank, 1, MPI_INT, xcomm->global_rank_to_local, 1, MPI_INT, xcomm->global_comm);
    MPI_Allgather(&node, 1, MPI_INT, xcomm->global_rank_to_node, 1, MPI_INT, xcomm->global_comm);

    xcomm->ordered_global_ranks = (int*)malloc(num_procs*sizeof(int));
    for (int i = 0; i < num_procs; i++)
    {
        int local = xcomm->global_rank_to_local[i];
        int node = xcomm->global_rank_to_node[i];
        xcomm->ordered_global_ranks[node*ppn + local] = i;
    }

    // Set xcomm variables
    MPI_Comm_size(xcomm->local_comm, &(xcomm->ppn));
    xcomm->num_nodes = ((num_procs-1) / xcomm->ppn) + 1;
    xcomm->rank_node = get_node(xcomm, rank);

    return MPI_SUCCESS;
}

void balancedBellmanFord(double* adjacencyMatrix, 
                         int* clusterMembership, 
                         int* centerRanks, 
                         double* distances, 
                         int* predecessors, 
                         int* numPredecessors, 
                         int* clusterSizes,
                         int tBFMax,
                         int numProcs,
                         int numClusters)
{
    int t = 0;
    int done = 1;
    while (t < tBFMax && !done)
    {
        for (int i = 0; i < numProcs; i++)
        {
            for (int j = 0; j < numProcs; j++)
            {
                int clusterISize = clusterSizes[clusterMembership[i]];
                int clusterJSize = clusterSizes[clusterMembership[j]];
                int swithClusters = 0;
                if (distances[i] + adjacencyMatrix[i * numProcs + j] < distances[j])
                    swithClusters = 1;
                else if (distances[i] + adjacencyMatrix[i * numProcs + j] == distances[j])
                {
                    if (clusterISize + 1 < clusterJSize)
                        swithClusters = true;
                }

                if (swithClusters)
                {
                    clusterSizes[clusterMembership[i]] = clusterISize + 1;
                    clusterSizes[clusterMembership[j]] = clusterJSize - 1;
                    clusterMembership[j] = clusterMembership[i];
                    distances[j] = distances[i] + adjacencyMatrix[i * numProcs + j];
                    numPredecessors[i] += 1;
                    numPredecessors[predecessors[j]] -= 1;
                    predecessors[j] = i;
                    done = false;
                }
            }
        }

        t += 1;
    }
}

void clusteredFloydWarshall(double* adjacencyMatrix, 
                            int* clusterMembership, 
                            int clusterSize,
                            int* cluster,
                            double** shortestPathDistances,
                            int** predecessors,
                            int numProcs)
{
    for (int i = 0; i < clusterSize; i++)
    {
        int start = cluster[i];
        for (int j = 0; j < clusterSize; j++)
        {
            int end = cluster[j];
            if (i == j)
            {
                shortestPathDistances[start][end] = 0;
                predecessors[start][start] = start;
            }
            else if (adjacencyMatrix[start * numProcs + end] > 0)
            {
                shortestPathDistances[start][end] = adjacencyMatrix[start * numProcs + end];
                predecessors[start][end] = start;
            }
            else
            {
                // this should never happen
                shortestPathDistances[start][end] = INFINITY;
                predecessors[start][end] = -1;
            }
        }
    }


    // // I don't think we actually need this second step, since we can assume each cluster is fully connected.
    // // However, for completeness, and the possibility that going through another process might be fast, if 
    // // that's reasonable
    // for (int k = 0; k < clusterSize; k++)
    // {
    //     for (int i = 0; i < clusterSize; i++)
    //     {
    //         for (int j = 0; j < clusterSize; j++)
    //         {
    //             if (i != k && j != k)
    //             {
    //                 double dist_ik = shortestPathDistances[cluster[i]][cluster[k]];
    //                 double dist_kj = shortestPathDistances[cluster[k]][cluster[j]];
    //                 if (dist_ik < dist_kj)
    //                 {
    //                     shortestPathDistances[cluster[i]][cluster[j]] = dist_ik + dist_kj;
    //                     predecessors[cluster[i]][cluster[j]] = predecessors[cluster[k], cluster[j]];
    //                 }
    //             }
    //         }
    //     }
    // }
}

void markUnavailable(int clusterIndex, 
                     int* clusterMembership, 
                     bool* clusterModifiable, 
                     double* adjacencyMatrix, 
                     int* clusterNodes,
                     int clusterSize,
                     int numProcs)
{
    clusterModifiable[clusterIndex] = false;
    for (int i = 0; i < clusterSize; i++)
    {
        for (int j = 0; j < numProcs; j++)
        {
            if (adjacencyMatrix[i * numProcs + j] > 0.0)
            {
                clusterModifiable[clusterMembership[j]] = false;
            }
        }
    }
}

void splitImprovementForCluster(int* clusterMembers,
                                double* shortestPathToCenters,
                                int numProcs,
                                double* shortestPathWithinCluster,
                                int clusterSize,
                                double *energyImprovement,
                                int* newCenter1,
                                int* newCenter2)
{
    *energyImprovement = INFINITY;
    for (int i = 0; i < clusterSize; i++)
    {
        for (int j = 0; j < clusterSize; j++)
        {
            double newEnergy = 0;
            for (int k = 0; k < clusterSize; k++)
            {
                if (shortestPathWithinCluster[i * numProcs + k] < shortestPathWithinCluster[j * numProcs + k])
                {
                    newEnergy += pow(shortestPathWithinCluster[i * numProcs + k], 2.0);
                }
                else
                {
                    newEnergy += pow(shortestPathWithinCluster[j * numProcs + k], 2.0);
                }
            }

            if (newEnergy < *energyImprovement)
            {
                *energyImprovement = newEnergy;
                *newCenter1 = i;
                *newCenter2 = j;
            }
        }
    }

    double energy = 0;
    for (int i = 0; i < clusterSize; i++)
    {
        energy += pow(shortestPathToCenters[i], 2.0);
    }

    energy -= *energyImprovement;
    *energyImprovement = energy;
}

double eliminationPenaltyForCluster(double *adjacencyMatrix, 
                                    int numProcs,
                                    int* clusterMembers,
                                    int clusterSize,
                                    double* shortestPathToCenters,
                                    double* shortestPathWithinCluster)
{
    double energyIncrease = 0;
    double currentEnergy = 0;
    for (int i = 0; i < clusterSize; i++)
    {
        currentEnergy += pow(shortestPathToCenters[i], 2.0);
        double minDistanceToCenter = INFINITY;
        for (int j = 0; j < clusterSize; j++)
        {
            for (int k = 0; k < numProcs; k++)
            {
                if (k != j && shortestPathToCenters[k] + adjacencyMatrix[k * numProcs + j] + shortestPathWithinCluster[j * numProcs + i] < minDistanceToCenter)
                {
                    minDistanceToCenter = shortestPathToCenters[k] + adjacencyMatrix[k * numProcs + j] + shortestPathWithinCluster[j * numProcs + i];
                }
            }
        }

        energyIncrease += minDistanceToCenter * minDistanceToCenter;
    }

    energyIncrease -= currentEnergy;

    return energyIncrease;
}

int compareArgSortable(const void* arg1, const void* arg2)
{
    ArgSortable* a1 = (ArgSortable*) arg1;
    ArgSortable* a2 = (ArgSortable*) arg2;
    if (a1->value < a2->value)
        return -1;
    else if ((a1->value > a2->value))
        return 1;
    else
        return 0;
}
 
void rebalance(double* adjacencyMatrix,
               int numProcs,
               int* clusterMembership,
               int** clusters,
               int* clusterSizes,
               int numClusters,
               int* clusterCenters,
               double* shortestPathToCenters,
               int* predecessors,
               double* shortestPathWithinCluster)
{
    ArgSortable* eliminationPenalties = (ArgSortable*) malloc(numClusters * sizeof(ArgSortable));
    ArgSortable* energyImprovements = (ArgSortable*) malloc(numClusters * sizeof(ArgSortable));
    int* newCenters1 = (int*) malloc(numClusters * sizeof(int));
    int* newCenters2 = (int*) malloc(numClusters * sizeof(int)); 
    bool* clusterModifiable = (bool*) malloc(numClusters * sizeof(bool));
    for (int a = 0; a < numClusters; a++)
    {
        eliminationPenalties[a].index = a;
        eliminationPenalties[a].value = eliminationPenaltyForCluster(adjacencyMatrix,
                                                               numProcs,
                                                               clusters[a],
                                                               clusterSizes[a],
                                                               shortestPathToCenters,
                                                               shortestPathWithinCluster);
        energyImprovements[a].index = a;
        splitImprovementForCluster(clusters[a],
                                   shortestPathToCenters,
                                   numProcs,
                                   shortestPathWithinCluster,
                                   clusterSizes[a],
                                   &energyImprovements[a].value,
                                   &newCenters1[a],
                                   &newCenters2[a]);
        clusterModifiable[a] = true;
    }

    qsort(eliminationPenalties, numClusters, sizeof(ArgSortable), compareArgSortable);
    qsort(energyImprovements, numClusters, sizeof(ArgSortable), compareArgSortable);

    int eliminateIndex = 0;
    int splitIndex = numClusters - 1;
    while (eliminateIndex < numClusters && splitIndex >= 0)
    {
        int eliminateCluster = eliminationPenalties[eliminateIndex].index;
        int splitCluster = energyImprovements[splitIndex].index;
        if (!clusterModifiable[eliminateCluster] || eliminateCluster == splitCluster)
        {
            eliminateCluster++;
        }
        else if (!clusterModifiable[splitCluster])
        {
            splitCluster--;
        }
        else if (energyImprovements[splitCluster].value < eliminationPenalties[eliminateCluster].value)
        {
            markUnavailable(eliminateCluster, 
                            clusterMembership, 
                            clusterModifiable, 
                            adjacencyMatrix, 
                            clusters[eliminateCluster], 
                            clusterSizes[eliminateCluster], 
                            numProcs);            
            markUnavailable(splitCluster, 
                            clusterMembership,
                            clusterModifiable,
                            adjacencyMatrix,
                            clusters[splitCluster],
                            clusterSizes[splitCluster],
                            numProcs);
            clusterCenters[eliminateIndex] = newCenters1[splitCluster];
            clusterCenters[splitCluster] = newCenters2[splitCluster];
        }
    } 
}

void centerNodes(double* adjacencyMatrix, 
                 int numProcs,
                 int* clusterMembership,
                 int numClusters,
                 int* clusterCenters,
                 double* shortestPathToCenter,
                 int* clusterCenterPredecessors,
                 int* numAsPredecessor,
                 double* shortestPathWithinCluster,
                 int* predecessors,
                 int** clusters,
                 int* clusterSizes)
{
    for (int a = 0; a < numClusters; a++)
    {
        double* sumSquaredDists = (double*) malloc(clusterSizes[a] * sizeof(double));
        double centerSumSquaredDists = 0.0;
        for (int i = 0; i < clusterSizes[a]; i++)
        {
            int procI = clusters[a][i];
            for (int j = 0; j < clusterSizes[a]; j++)
            {
                int procJ = clusters[a][j];
                if (shortestPathToCenter[i * numProcs + j] > 0.0)
                {
                    sumSquaredDists[i] += pow(shortestPathToCenter[procI * numProcs + procJ], 2.0);
                }
            }

            if (procI == clusterCenters[a])
                centerSumSquaredDists = sumSquaredDists[i];

            int potentialCenter = clusterCenters[a];
            for (int j = 0; j < clusterSizes[a]; j++)
            {
                if (sumSquaredDists[j] < centerSumSquaredDists)
                {
                    potentialCenter = j;
                    centerSumSquaredDists = sumSquaredDists[j];
                }   
            }

            if (potentialCenter != clusterCenters[a])
            {
                clusterCenters[a] = potentialCenter;
                for (int j = 0; j < clusterSizes[a]; j++)
                {
                    numAsPredecessor[clusters[a][j]] = 0;
                }

                for (int j = 0; j < clusterSizes[a]; j++)
                {
                    shortestPathToCenter[j] = shortestPathWithinCluster[potentialCenter * numProcs + j];
                    clusterCenterPredecessors[j] = predecessors[i * numProcs + j];
                    numAsPredecessor[clusterCenterPredecessors[j]] += 1;
                }
            }
        }

        free(sumSquaredDists);
    }
}

// void balancedLloydClustering(double* adjancencyMatrix, int* centerRanks, int numProcs, int numClusters, int tMax, int tBFMax)
// {
//     // balanced initialization
//     int* clusterMembership = (int*) malloc(numProcs * sizeof(int));
//     double* distances = (double*) malloc(numProcs * sizeof(double));
//     int** predecessors = (int**) malloc(numProcs * sizeof(int*));
//     int* numPredecessors = (int*) malloc(numProcs * sizeof(int));
//     for (int i = 0; i < numProcs; i++)
//     {
//         clusterMembership[i] = 0;
//         distances[i] = INFINITY;
//         predecessors[i] = (int*) calloc(numProcs, sizeof(int));
//         numPredecessors = 0;
//     }

//     int* clusterSizes = (int*) malloc(numClusters * sizeof(int));
//     memset(clusterSizes, 1, numClusters);
//     srand(time(NULL));

//     for (int clusterIndex = 0; clusterIndex < numClusters; clusterIndex++)
//     {
//         centerRanks[clusterIndex] = rand() % numProcs;
//         int nodeIndex = clusterCenters[clusterIndex];
//         distances[nodeIndex] = 0;
//         clusterMembership[nodeIndex] = clusterIndex;
//         predecessors[nodeIndex] = nodeIndex;
//         numPredecessors[nodeIndex] = 1;
//     }    

//     int t = 0;
//     while (t < tMax) // need to add other condition (m, c, d, p, n, s don't change)
//     {
//         balancedBellmanFord(adjancencyMatrix, 
//                             clusterMembership,
//                             centerRanks,
//                             distances,
//                             predecessors,   
//                             numPredecessors,
//                             clusterSizes,
//                             tBFMax,
//                             numProcs,
//                             numClusters);
        
//         for (int i = 0; i < numClusters; i++)
//         {
//             clusteredFloydWarshall(adjancencyMatrix, 
//                                    clusterMembership,
//                                    clusterSizes[i],
//                                    cluster[i],
//                                    distances,
//                                    predecessors,
//                                    numProcs);
//         }
        
//     }
// }

double* network_discovery(MPIX_Comm* xcomm, int size, int tag, int num_iterations)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);
    
    double* averageDistances = (double*) calloc(num_procs, sizeof(double));

    char* send_buffer = (char*) malloc(num_procs * size * sizeof(char));
    char* recv_buffer = (char*) malloc(num_procs * size * sizeof(char));

    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;
    averageDistances[rank] = 0.0;
    for (int i = 0; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;

        send_pos = send_proc * sizeof(char);
        recv_pos = recv_proc * sizeof(char); 

        // Warm up
        MPI_Sendrecv(send_buffer + send_pos,
                     1,
                     MPI_CHAR, 
                     send_proc,
                     tag,
                     recv_buffer + recv_pos,
                     1,
                     MPI_CHAR,
                     recv_proc,
                     tag,
                     xcomm->global_comm,
                     &status);

        MPI_Sendrecv(send_buffer + recv_pos,
                     1,
                     MPI_CHAR,
                     recv_proc,
                     tag,
                     recv_buffer + send_pos,
                     1,
                     MPI_CHAR,
                     send_proc,
                     tag,
                     xcomm->global_comm, 
                     &status);

        double t0 = MPI_Wtime();
        for (int i = 0; i < num_iterations; i++)
        {
            MPI_Sendrecv(send_buffer + send_pos,
                        1,
                        MPI_CHAR, 
                        send_proc,
                        tag,
                        recv_buffer + recv_pos,
                        1,
                        MPI_CHAR,
                        recv_proc,
                        tag,
                        xcomm->global_comm,
                        &status);

            MPI_Sendrecv(send_buffer + recv_pos,
                        1,
                        MPI_CHAR,
                        recv_proc,
                        tag,
                        recv_buffer + send_pos,
                        1,
                        MPI_CHAR,
                        send_proc,
                        tag,
                        xcomm->global_comm, 
                        &status);
        }

        averageDistances[send_proc] = averageDistances[recv_proc] = (MPI_Wtime() - t0) / (2. * (double) num_iterations);
    }

    double* adjacencyMatrix = (double*) malloc(num_procs * num_procs * sizeof(double));
    MPI_Allgather(averageDistances, num_procs, MPI_DOUBLE, adjacencyMatrix, num_procs, MPI_DOUBLE, xcomm->global_comm);

    return adjacencyMatrix;
}

int MPIX_Comm_topo_cluster_init(MPIX_Comm* xcomm)
{
    int tag;
    MPIX_Comm_tag(xcomm, &tag);
    double* adjacencyMatrix = network_discovery(xcomm, 2, tag, 1);
    return MPI_SUCCESS;
}

int MPIX_Comm_leader_init(MPIX_Comm* xcomm, int procs_per_leader)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    MPI_Comm_split(xcomm->global_comm,
        rank / procs_per_leader,
        rank,
        &(xcomm->leader_comm));

    int leader_rank;
    MPI_Comm_rank(xcomm->leader_comm, &leader_rank);

    MPI_Comm_split(xcomm->global_comm,
        leader_rank,
        rank,
        &(xcomm->leader_group_comm));

    if (xcomm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(xcomm);

    MPI_Comm_split(xcomm->local_comm,
        leader_rank,
        rank,
        &(xcomm->leader_local_comm));

    return MPI_SUCCESS;
}

int MPIX_Comm_device_init(MPIX_Comm* xcomm)
{
#ifdef GPU
    if (xcomm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(xcomm);

    int local_rank, ierr;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    ierr = gpuGetDeviceCount(&(xcomm->gpus_per_node));
    gpu_check(ierr);
    if (xcomm->gpus_per_node)
    {
        xcomm->rank_gpu = local_rank;
        ierr = gpuStreamCreate(&(xcomm->proc_stream));
        gpu_check(ierr);
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
    xcomm->statuses = (MPI_Status*)realloc(xcomm->statuses, n*sizeof(MPI_Status));

    return MPI_SUCCESS;
}

int MPIX_Comm_tag(MPIX_Comm* xcomm, int* tag)
{
    *tag = xcomm->tag;
    xcomm->tag = ((xcomm->tag + 1 ) % xcomm->max_tag);

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
    MPIX_Comm_leader_free(xcomm);
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

    if (xcomm->global_rank_to_local != NULL)
        free(xcomm->global_rank_to_local);
    if (xcomm->global_rank_to_node != NULL)
        free(xcomm->global_rank_to_node);
    if (xcomm->ordered_global_ranks != NULL)
        free(xcomm->ordered_global_ranks); 

    return MPI_SUCCESS;
}

int MPIX_Comm_leader_free(MPIX_Comm* xcomm)
{
    if (xcomm->leader_comm != MPI_COMM_NULL)
      MPI_Comm_free(&(xcomm->leader_comm));
    if (xcomm->leader_group_comm != MPI_COMM_NULL)
       MPI_Comm_free(&(xcomm->leader_group_comm));
    if (xcomm->leader_local_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(xcomm->leader_local_comm));

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
    int ierr = gpuSuccess;
    if (xcomm->gpus_per_node)
        ierr = gpuStreamDestroy(xcomm->proc_stream);
    gpu_check(ierr);
#endif

    return MPI_SUCCESS;
}




/****  Topology Functions   ****/
int get_node(const MPIX_Comm* data, const int proc)
{
    return data->global_rank_to_node[proc]; 
}

int get_local_proc(const MPIX_Comm* data, const int proc)
{
    return data->global_rank_to_local[proc];
}

int get_global_proc(const MPIX_Comm* data, const int node, const int local_proc)
{
    return data->ordered_global_ranks[local_proc + (node * data->ppn)];
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



    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_split(xcomm->global_comm,
        local_rank,
        rank,
        &(xcomm->group_comm));

    int node;
    MPI_Comm_rank(xcomm->group_comm, &node);


    if (xcomm->global_rank_to_local == NULL)
        xcomm->global_rank_to_local = (int*)malloc(num_procs*sizeof(int));

    if (xcomm->global_rank_to_node == NULL)
        xcomm->global_rank_to_node = (int*)malloc(num_procs*sizeof(int));

    MPI_Allgather(&local_rank, 1, MPI_INT, xcomm->global_rank_to_local, 1, MPI_INT, xcomm->global_comm);
    MPI_Allgather(&node, 1, MPI_INT, xcomm->global_rank_to_node, 1, MPI_INT, xcomm->global_comm);

    if (xcomm->ordered_global_ranks == NULL)
        xcomm->ordered_global_ranks = (int*)malloc(num_procs*sizeof(int));

    for (int i = 0; i < num_procs; i++)
    {
        int local = xcomm->global_rank_to_local[i];
        int node = xcomm->global_rank_to_node[i];
        xcomm->ordered_global_ranks[node*ppn + local] = i;
    }

    MPI_Comm_size(xcomm->local_comm, &(xcomm->ppn));
    xcomm->num_nodes = ((num_procs - 1) / xcomm->ppn) + 1;
    xcomm->rank_node = get_node(xcomm, rank);
}

