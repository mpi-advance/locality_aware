#include "mpi_advance.h"

#include <algorithm>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>

int testArrays(double* array1, double* array2, int size, double tolerance)
{
    int numErrors = 0;
    for (int i = 0; i < size; i++)
    {
        if (fabs(array1[i] - array2[i]) > tolerance) 
        {
            numErrors++;
            fprintf(stderr, "Index: %d, array1: %f, array2: %f are not equal\n", i, array1[i], array2[i]);
        }
    }

    return numErrors;
}

int testArrays(int* array1, int* array2, int size)
{
    int numErrors = 0;
    for (int i = 0; i < size; i++)
    {
        if (array1[i] != array2[i])
        {
            fprintf(stderr, "Index: %d, array1: %d, array2: %d are not equal\n", i, array1[i], array2[i]);
        }
     }

     return numErrors;
}

int main()
{
    printf("\n=== test_clustering START ===\n");
    fflush(stdout);
    
    /*
     *  This test case is borrowed from pyAMG
     *  Unit length edges
     *
     *         0 --- 3
     *         | \ / |
     *         | / \ |
     *         1 --- 4       
     *         | \ / |
     *         | / \ |
     *         2 --- 5
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

    int clusterCenters[2] = {0, 5};
    std::vector<double> shortestPathToCenter(numProcs, INFINITY);
    std::fill(clusterMembership.begin(), clusterMembership.end(), -1);
    std::vector<int> predecessorInCluster(numProcs, -1);
    std::vector<int> numAsPredecessor(numProcs, 0);
    clusterSizes[0] = 1;
    clusterSizes[1] = 1;
    for (int i = 0; i < numClusters; i++) 
    {
        int center = clusterCenters[i];
        shortestPathToCenter[center] = 0;
        clusterMembership[center] = i;
    }

    std::vector<int> expectedMembership = {0, 0, 1, 0, 1, 1};
    std::vector<double> expectedShortestPathToCenter = {0, 1, 1, 1, 1, 0};
    std::vector<int> expectedClusterSizes = {3, 3};
    bool changed = balancedBellmanFord(adjacencyMatrix, 
                                       clusterMembership.data(),
                                       clusterCenters,
                                       shortestPathToCenter.data(),
                                       predecessorInCluster.data(),
                                       numAsPredecessor.data(),
                                       clusterSizes.data(),
                                       numProcs,
                                       numClusters,
                                       10000);
    printf("BalancedBellmanFord changed: %s\n", (changed ? "True" : "False"));

    int numErrors = 0;
    printf("Testing shortest path to center\n");
    numErrors += testArrays(shortestPathToCenter.data(), expectedShortestPathToCenter.data(), expectedShortestPathToCenter.size(), 0.0001);
    printf("Testing clusterMembership\n");
    numErrors += testArrays(clusterMembership.data(), expectedMembership.data(), expectedMembership.size());
    printf("Testing cluster sizes\n");
    numErrors += testArrays(clusterSizes.data(), expectedClusterSizes.data(), expectedClusterSizes.size());
    // To test center nodes, we'll the clusters a bit.
    // cluster 0: 0, 1, 2
    // cluster 1: 3, 4, 5
    // centers: 0, 5
    // predecessors: -1 0 1 4 5 -1
    // The center node will not be connected to one of the nodes, so centerNodes
    // will be forced to swap it
    /*
     *         0 --- 3
     *         | \ / |
     *         | / \ |
     *         1 --- 4       
     *         | \ / |
     *         | / \ |
     *         2 --- 5
     */

    clusters[0][0] = 0;
    clusters[0][1] = 1;
    clusters[0][2] = 2;
    clusters[1][0] = 3;
    clusters[1][1] = 4;
    clusters[1][2] = 5;

    clusterMembership = {0, 0, 0, 1, 1, 1};
    shortestPathDistances = {0, 1, 2, 2, 1, 0};
    predecessorInCluster = {0, 0, 1, 4, 5, 5};
    numAsPredecessor = {1, 1, 0, 0, 1, 1};
    std::vector<double> shortestPathWithinCluster(numProcs * numProcs, INFINITY);

    // set the shortest paths for clusters.  Use the same math for indexing for clarity
    // shortestPathWithinCluster[0 * numProcs + 0] = 0;
    // shortestPathWithinCluster[0 * numProcs + 1] = 1;
    // shortestPathWithinCluster[0 * numProcs + 2] = 2;
    // shortestPathWithinCluster[1 * numProcs + 0] = 1;
    // shortestPathWithinCluster[1 * numProcs + 1] = 0;
    // shortestPathWithinCluster[1 * numProcs + 2] = 1;
    // shortestPathWithinCluster[2 * numProcs + 0] = 2;
    // shortestPathWithinCluster[2 * numProcs + 1] = 1;
    // shortestPathWithinCluster[2 * numProcs + 2] = 0;
    // shortestPathWithinCluster[3 * numProcs + 3] = 0;
    // shortestPathWithinCluster[3 * numProcs + 4] = 1;
    // shortestPathWithinCluster[3 * numProcs + 5] = 2;
    // shortestPathWithinCluster[4 * numProcs + 3] = 1;
    // shortestPathWithinCluster[4 * numProcs + 4] = 0;
    // shortestPathWithinCluster[4 * numProcs + 5] = 1;
    // shortestPathWithinCluster[5 * numProcs + 3] = 2;
    // shortestPathWithinCluster[5 * numProcs + 4] = 1;
    // shortestPathWithinCluster[5 * numProcs + 5] = 0;

    std::fill(predecessors.begin(), predecessors.end(), -1);

    clusteredFloydWarshall(adjacencyMatrix,
                           clusterMembership.data(),
                           clusterSizes.data(),
                           clusters, 
                           numClusters,
                           shortestPathWithinCluster.data(),
                           predecessors.data(),
                           numProcs);

    //predecessors for the shortest path in the clusters
    // predecessors[0 * numProcs + 1] = 
    int expectedClusterCenters[2] = {1, 4};
    changed = centerNodes(adjacencyMatrix,
                          numProcs,
                          clusterMembership.data(),
                          numClusters,
                          clusterCenters,
                          shortestPathDistances.data(),
                          predecessorInCluster.data(),
                          numAsPredecessor.data(),
                          shortestPathWithinCluster.data(),
                          predecessors.data(),
                          clusters,
                          clusterSizes.data());

    numErrors += testArrays(clusterCenters, expectedClusterCenters, 2);

    if (numErrors == 0) 
    {
        printf("=== test_clustering PASS ===\n");
    }
    else
    {
        printf("=== test_cluster FAIL (%d errors) ===\n", numErrors);
    }
}