#include "clustering.h"
#include <math.h>

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
