#ifndef MPIL_TOPO_H
#define MPIL_TOPO_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Struct to hold topology weights
    @details
		Structure containing the same arguments as MPI_Dist_graph_create_adjacent.
		Information is cached to avoiding to rebuild communicators as often. 
		See MPI standard: MPI_Dist_graph_create_adjacent for more details. 
**/
typedef struct _MPIL_Topo
{
	/** @brief size of sources and source weights **/
    int indegree;
	/** @brief rank for which the process is a a destination **/
    int* sources;
	/** weights of the edges into the calling process**/
    int* sourceweights; 
	/** @brief size of destinations and destweights **/
    int outdegree;      
	/** @brief ranks of processes for which the calling process is a source**/
    int* destinations;  
	/** @brief weights of the edges outof the calling process**/
    int* destweights;   
	/** @brief the ranks may be reordered (true if 0) false otherwise**/
    int reorder;        
} MPIL_Topo;

#ifdef __cplusplus
}
#endif

#endif