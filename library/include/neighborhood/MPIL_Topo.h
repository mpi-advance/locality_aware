#ifndef MPIL_TOPO_H
#define MPIL_TOPO_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _MPIL_Topo
{
    int indegree;
    int* sources;
    int* sourceweights;
    int outdegree;
    int* destinations;
    int* destweights;
    int reorder;
} MPIL_Topo;

#ifdef __cplusplus
}
#endif

#endif