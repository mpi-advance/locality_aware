#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "mpix_comm.h"


double* network_discovery(MPIX_Comm* xcomm, int size, int tag, int num_iterations); // This is only exposed temporarily for some scaling studies

void balancedLloydClustering(double* adjacencyMatrix,
                             int** clusterCenters,
                             int** clusterMembership,
                             int maxIterations,
                             int maxBellmanFordIterations,
                             int numProcs,
                             int numClusters);

void clusteredFloydWarshall(double* adjacencyMatrix, 
                            int* clusterMembership, 
                            int* clusterSizes,
                            int** clusters,
                            int numClusters,
                            double* shortestPathDistances,
                            int* predecessors,
                            int numProcs);

#endif // CLUSTERING_H