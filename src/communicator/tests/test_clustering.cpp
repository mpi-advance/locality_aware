#include "mpi_advance.h"

#include <algorithm>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>

void testArrays(double* array1, double* array2, int size, double tolerance)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(array1[i] - array2[i]) > tolerance) 
        {
            fprintf(stderr, "Index: %d, array1: %f, array2: %f are not equal\n", i, array1[i], array2[i]);
        }
    }
}

void testArrays(int* array1, int* array2, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (array1[i] != array2[i])
        {
            fprintf(stderr, "Index: %d, array1: %d, array2: %d are not equal\n", i, array1[i], array2[i]);
        }
     }
}

int main()
{
    printf("\n=== test_clustering START ===\n");
    fflush(stdout);
    
    /*
     *  Unit length edges
     *
     *         0                3
     * 
     * 
     *         1                4       
     * 
     * 
     *         2                5
     *
     */
    double adjacencyMatrix[36] = {
                                    0, 1, 0, 1, 1, 0,
                                    1, 0, 1, 1, 1, 1,
                                    0, 1, 0, 0, 1, 1,
                                    1, 1, 0, 0, 1, 0,
                                    1, 1, 1, 1, 0, 1,
                                    0, 1, 1, 0, 1, 0
                                 };

    std::vector<int> clusterMembership = {0, 0, 1, 0, 1, 1};
    std::vector<int> clusterSizes = {3, 3};
    int numClusters = 2;
    int** clusters = (int**) malloc(2 * sizeof(int*));
    clusters[0] = (int*) malloc(3 * sizeof(int));
    clusters[0][0] = 0;
    clusters[0][1] = 1;
    clusters[0][2] = 3;
    clusters[1] = (int*) malloc(3 * sizeof(int));
    clusters[1][0] = 5;
    clusters[1][1] = 2;
    clusters[1][2] = 4;
    int numProcs = 6;
    std::vector<double> shortestPathDistances(numProcs * numProcs, 0.0);
    double expectedShortestPathDistances[36] = {0, 1, 0, 1, 0, 0,
                                                1, 0, 0, 1, 0, 0,
                                                0, 0, 0, 0, 1, 1,
                                                1, 1, 0, 0, 0, 0,
                                                0, 0, 1, 0, 0, 1,
                                                0, 0, 1, 0, 1, 0};

    std::vector<int> predecessors(numProcs * numProcs, -1);
    int expectedPredecessors[36] = {0, 0, -1, 0, -1, -1,
                                    1, 1, -1, 1, -1, -1,
                                    -1, -1, 2, -1, 2, 2,
                                    3, 3, -1, 3, -1, -1,
                                    -1, -1, 4, -1, 4, 4,
                                    -1, -1, 5, -1, 5, 5};

    clusteredFloydWarshall(adjacencyMatrix, 
                           clusterMembership.data(),
                           clusterSizes.data(),
                           clusters,
                           numClusters, 
                           shortestPathDistances.data(),
                           predecessors.data(),
                           numProcs);
    
    printf("Testing shortest path distances\n");
    testArrays(shortestPathDistances.data(), expectedShortestPathDistances, numProcs * numProcs, 0.00001);

    printf("Testing predecessors\n");
    testArrays(predecessors.data(), expectedPredecessors, numProcs * numProcs);

    printf("[DEBUG] clusteredFloydWarshall completed successfully\n");
    fflush(stdout);
    printf("[DEBUG] Calling balancedBellmanFord...\n");
    fflush(stdout);

    int centerNodes[2] = {0, 5};
    std::vector<double> shortestPathToCenter(numProcs, INFINITY);
    std::fill(clusterMembership.begin(), clusterMembership.end(), -1);
    std::vector<int> predecessorInCluster(numProcs, -1);
    std::vector<int> numAsPredecessor(numProcs, 0);
    clusterSizes[0] = 1;
    clusterSizes[1] = 1;
    int clusterCenters[2] = {0, 5};
    for (int i = 0; i < numClusters; i++) 
    {
        int center = clusterCenters[i];
        shortestPathToCenter[center] = 0;
        clusterMembership[center] = i;
    }

    std::vector<int> expectedMembership = {0, 0, 1, 0, 1, 1};
    std::vector<double> expectedShortestPathToCenter = {0, 1, 1, 1, 1, 0};

    bool changed = balancedBellmanFord(adjacencyMatrix, 
                                    clusterMembership.data(),
                                    centerNodes,
                                    shortestPathToCenter.data(),
                                    predecessorInCluster.data(),
                                    numAsPredecessor.data(),
                                    clusterSizes.data(),
                                    numProcs,
                                    numClusters,
                                    10000);
    printf("BalancedBellmanFord changed: %s\n", (changed ? "True" : "False"));

    printf("Testing shortest path to center\n");
    testArrays(shortestPathToCenter.data(), expectedShortestPathToCenter.data(), expectedShortestPathToCenter.size(), 0.0001);
    printf("Testing clusterMembership\n");
    testArrays(clusterMembership.data(), expectedMembership.data(), expectedMembership.size());
    
    printf("=== test_clustering PASS ===\n");
    fflush(stdout);
}