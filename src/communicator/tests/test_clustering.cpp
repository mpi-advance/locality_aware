#include "../clustering.h"

void testClusteredFloydWarshallFullyConnected(double* adjacencyMatrix)
{
    int clusterMembership[6] = {0, 0, 0, 1, 1, 1};
    int clusterSizes[2] = {3, 3};
    int** clusters = (int**) malloc(2 * sizeof(int*));
    clusters[0] = (int*) malloc(3 * sizeof(int));
    clusters[0][0] = 0;
    clusters[0][1] = 1;
    clusters[0][2] = 2;
    
    clusters[1] = (int*) malloc(3 * sizeof(int));
    clusters[1][0] = 3;
    clusters[1][1] = 4;
    clusters[1][2] = 5;

    double* shortestPathDistances = (double*) calloc(36, sizeof(double));
    int* predecessors = (int*) malloc(36 * sizeof(int));
    for (int i = 0; i < 36; i++)
    {
        predecessors[i] = -1;
    }

    clusteredFloydWarshall(adjacencyMatrix, clusterMembership, clusterSizes, clusters, 2, shortestPathDistances, predecessors, 6);
    // how to handle validation outside of google test?
}

int main(int argc, char* argv)
{
    /*
     *
     *
     *         0                3
     * 
     * 
     *         1                4       
     * 
     * 
     *         2                5
     * 
     * 
     * Clusters: (0, 1, 2), (3, 4, 5)
     * Centers: 1, 4 
     */
    double adjacencyMatrix[36] = { 0, 1, 2, 6, 6, 6,
                                   1, 0, 1, 6, 3, 6,
                                   2, 1, 0, 6, 6, 6,
                                   6, 6, 6, 0, 1, 2,
                                   6, 3, 6, 1, 0, 1,
                                   6, 6, 6, 2, 1, 0 };

    testClusteredFloydWarshallFullyConnected(adjacencyMatrix);
    // balancedLloydClustering(adjacencyMatrix, &clusterCenters, &clusterMembership, 10, 10, 6, 2);

    // for (int i = 0; i < 6; i++)
    // {
    //     printf("Rank: %d, cluster: %d\n", i, clusterMembership[i]);
    // }

    // for (int i = 0; i < 2; i++)
    // {
    //     printf("Cluster: %d, center: %d\n", i, clusterCenters[i]);
    // }
}