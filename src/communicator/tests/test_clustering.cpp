#include "mpi_advance.h"

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

    int clusterMembership[6] = {0, 0, 1, 0, 1, 1};
    int clusterSizes[2] = {3, 3};
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
                           clusterMembership,
                           clusterSizes,
                           clusters,
                           numClusters, 
                           shortestPathDistances.data(),
                           predecessors.data(),
                           numProcs);
    
    printf("Testing shortest path distances\n");
    testArrays(shortestPathDistances.data(), expectedShortestPathDistances, numProcs * numProcs, 0.00001);

    printf("Testing predecessors\n");
    testArrays(predecessors.data(), expectedPredecessors, numProcs * numProcs);
}