#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "mpix_comm.h"


double* network_discovery(MPIX_Comm* xcomm, int size, int tag, int num_iterations); // This is only exposed temporarily for some scaling studies

bool balancedBellmanFord(double* adjacencyMatrix, 
                         int* clusterMembership, 
                         int* centerNodes, 
                         double* shortestPathToCenter,
                         int* predecessorsInCluster, 
                         int* numAsPredecessor, 
                         int* clusterSizes,
                         int numProcs,
                         int numClusters,
                         int maxIterations);

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

// exposed for testing only
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
                 int* clusterSizes);

#endif // CLUSTERING_H