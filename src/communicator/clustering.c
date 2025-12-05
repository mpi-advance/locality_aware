#include "clustering.h"
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>

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
    for (int i = 1; i < num_procs; i++)
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

        double t0 = MPI_Wtime();
        for (int j = 0; j < num_iterations; j++)
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
        }

        averageDistances[send_proc] = (MPI_Wtime() - t0) / (2. * (double) num_iterations);

        // MPI_Sendrecv(send_buffer + recv_pos,
        //              1,
        //              MPI_CHAR,
        //              recv_proc,
        //              tag,
        //              recv_buffer + send_pos,
        //              1,
        //              MPI_CHAR,
        //              send_proc,
        //              tag,
        //              xcomm->global_comm, 
        //              &status);

        // t0 = MPI_Wtime();
        // for (int j = 0; j < num_iterations; j++)
        // {
        //     MPI_Sendrecv(send_buffer + recv_pos,
        //                 1,
        //                 MPI_CHAR,
        //                 recv_proc,
        //                 tag,
        //                 recv_buffer + send_pos,
        //                 1,
        //                 MPI_CHAR,
        //                 send_proc,
        //                 tag,
        //                 xcomm->global_comm, 
        //                 &status);
        // }

        // averageDistances[recv_proc] = (MPI_Wtime() - t0) / (2. * (double) num_iterations);

    }

    double* adjacencyMatrix = (double*) malloc(num_procs * num_procs * sizeof(double));
    MPI_Allgather(averageDistances, num_procs, MPI_DOUBLE, adjacencyMatrix, num_procs, MPI_DOUBLE, xcomm->global_comm);

    return adjacencyMatrix;
}

bool balancedBellmanFord(double* adjacencyMatrix, 
                         int* clusterMembership, 
                         int* centerNodes, 
                         double* shortestPathToCenter,
                         int* predecessorsInCluster, 
                         int* numAsPredecessor, 
                         int* clusterSizes,
                         int numProcs,
                         int numClusters,
                         int maxIterations)
{
    int changed = false;

    int t = 0;
    bool done;
    do {
        done = true;
        for (int i = 0; i < numProcs; i++)
        {
            for (int j = 0; j < numProcs; j++)
            {
                int iClusterSize= clusterSizes[clusterMembership[i]];
                int jClusterSize = clusterSizes[clusterMembership[j]];
                bool shouldSwitch = false;
                if (shortestPathToCenter[i] + adjacencyMatrix[i * numProcs + j] < shortestPathToCenter[j])
                    shouldSwitch = true;
                
                // the algorithm says similar, should we use approximate equals here? what would the tolerance be?
                if (shortestPathToCenter[i] + adjacencyMatrix[i * numProcs + j] == shortestPathToCenter[j]) 
                {
                    if (iClusterSize + 1 < jClusterSize && numAsPredecessor[j] == 0)
                        shouldSwitch = true;
                }

                if (shouldSwitch)
                {
                    clusterSizes[clusterMembership[i]] = iClusterSize + 1;
                    clusterSizes[clusterMembership[j]] = jClusterSize - 1;
                    clusterMembership[j] = clusterMembership[i];
                    shortestPathToCenter[j] = shortestPathToCenter[i] + adjacencyMatrix[i * numProcs + j];
                    numAsPredecessor[i] += 1;
                    numAsPredecessor[predecessorsInCluster[j]] -= 1;
                    predecessorsInCluster[j] = i;
                    changed = true;
                    done = false;
                }
            }
        }

        t++;
    } 
    while (t < maxIterations && !done);
    return changed;
}

void clusteredFloydWarshall(double* adjacencyMatrix, 
                            int* clusterMembership, 
                            int* clusterSizes,
                            int** clusters,
                            int numClusters,
                            double* shortestPathDistances,
                            int* predecessors,
                            int numProcs)
{
    for (int a = 0; a < numClusters; a++)
    {
        for (int i = 0; i < clusterSizes[a]; i++)
        {
            int start = clusters[a][i];
            for (int j = 0; j < clusterSizes[a]; j++)
            {
                int end = clusters[a][j];
                if (adjacencyMatrix[start * numProcs + end] > 0)
                {
                    shortestPathDistances[start * numProcs + end] = adjacencyMatrix[start * numProcs + end];
                    predecessors[start * numProcs + end] = start;
                }
                else if (start == end)
                {
                    // if the adjacency matrix properly sets the distance to self as 0, then this is 
                    // the same as the previous case.
                    shortestPathDistances[start * numProcs + start] = 0.0;
                    predecessors[start * numProcs + end] = start;
                }
                else
                {
                    // this should never happen on a fully connected graph
                    shortestPathDistances[start * numProcs + end] = INFINITY;
                    predecessors[start * numProcs + end] = -1;
                }
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
    //                 double dist_ik = shortestPathDistances[cluster[i] * numProcs + cluster[k]];
    //                 double dist_kj = shortestPathDistances[cluster[k] * numProcs + cluster[j]];
    //                 if (dist_ik < dist_kj)
    //                 {
    //                     shortestPathDistances[cluster[i] * numProcs + cluster[j]] = dist_ik + dist_kj;
    //                     predecessors[cluster[i] * numProcs + cluster[j]] = predecessors[cluster[k] * numProcs + cluster[j]];
    //                 }
    //             }
    //         }
    //     }
    // }
}

bool centerNodes(double* adjacencyMatrix, 
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
    bool changed = false;
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

                changed = true;
            }
        }

        free(sumSquaredDists);
    }
    
    return changed;
}

void balancedLloydClustering(double* adjacencyMatrix,
                             int** clusterCenters,
                             int** clusterMembership,
                             int maxIterations,
                             int maxBellmanFordIterations,
                             int numProcs,
                             int numClusters)
{
    // balanced initialization
    *clusterMembership = (int*) malloc(numProcs * sizeof(int));
    double* shortestPathToCenter = (double*) malloc(numProcs * sizeof(double));
    int* predecessorInCluster = (int*) malloc(numProcs * sizeof(int));
    int* numAsPredecessor = (int*) malloc(numProcs * sizeof(int));
    for (int i = 0; i < numProcs; i++)
    {
        clusterMembership[0][i] = -1;
        shortestPathToCenter[i] = INFINITY;
        predecessorInCluster[i] = -1;
        numAsPredecessor[i] = 0;
    }

    int* clusterSizes = (int*) malloc(numClusters * sizeof(int));
    *clusterCenters = (int*) malloc(numClusters * sizeof(int));
    srand(time(NULL));
    for (int a = 0; a < numClusters; a++)
    {
        clusterCenters[0][a] = rand() % numProcs;
        int nodeIndex = clusterCenters[0][a];
        shortestPathToCenter[a] = 0;
        clusterMembership[0][nodeIndex] = a;
        predecessorInCluster[nodeIndex] = nodeIndex;
        numAsPredecessor[nodeIndex] = 1;
    }

    int iteration = 0;
    bool changed = true;
    do 
    {
        printf("Iteration: %d\n", iteration);
        changed = balancedBellmanFord(adjacencyMatrix, 
                                      *clusterMembership, 
                                      *clusterCenters, 
                                      shortestPathToCenter,
                                      predecessorInCluster,
                                      numAsPredecessor, 
                                      clusterSizes,
                                      numProcs,
                                      numClusters,
                                      maxIterations);

        int** clusters = (int**) malloc(numClusters * sizeof(int*));
        int positionInCluster[numClusters];
        for (int a = 0; a < numClusters; a++)
        {
            clusters[a] = (int*) malloc(clusterSizes[a] * sizeof(int));
            positionInCluster[a] = 0;
        }

        for (int i = 0; i < numProcs; i++)
        {
            int cluster = clusterMembership[0][i];
            printf("i: %d, cluster: %d, positionInCluster: %d\n", i, cluster, positionInCluster[cluster]);
            // clusters[cluster][positionInCluster[cluster]] = i;
            // positionInCluster[cluster]++;
        }

        // printf("Calling clustered Floyd Warhsall\n");
        // clusteredFloydWarshall(adjacencyMatrix, 
        //                        *clusterMembership, 
        //                        clusterSizes,
        //                        clusters,
        //                        numClusters,
        //                        shortestPathToCenter,
        //                        predecessorInCluster,
        //                        numProcs);

        // printf("Center nodes\n");
        // changed |= centerNodes(adjacencyMatrix, 
        //                        numProcs,
        //                        *clusterMembership,
        //                        numClusters,
        //                        *clusterCenters,
        //                        shortestPathToCenter,
        //                        predecessorInCluster,
        //                        numAsPredecessor,
        //                        shortestPathToCenter,
        //                        predecessorInCluster,
        //                        clusters,
        //                        clusterSizes);

        iteration++;
    } while (iteration < maxIterations && changed);
}

/////////////////////////////////////////////////////////
// BEGIN REBALANCING CODE - FOR FUTURE IMPLEMENTATIONS //
/////////////////////////////////////////////////////////
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
//////////////////////////
// END REBALANCING CODE //
/////////////////////////