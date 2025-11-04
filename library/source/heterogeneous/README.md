# Overview

# Building Instructions

## Prequistites and Dependencies

## Building

### Build options

# Using the Library

## Linking

## Testing

## API

### Basic Collectives

### Neighborhood Collectives

### Modified MPI Functions

### Supporting Functions

### Structs and Classes. 
The library provides the following protected structs. 

- MPIL_Comm
- MPIL_Info
- MPIL_Topo
- MPIL_Request

Structs and classes provided by the library should be created and accessed
through the supplied API calls. 
For functions provided by the library the provided functions should be used in place of the standard MPI structs of the similar name. 

# Repository Layout

# Acknowledgements
This work has been partially funded by ...

############################################################

# Locality-Aware MPI
This repository performs locality-aware optimizations for standard MPI collectives as well as neighborhood collectives.

## Collective Optimizations
The collective optimizations are within the folder src/collective.

### Allgather :
The file allgather.c contains methods for performing the bruck allgather, the ring allgather, and point-to-point communication (all processes perform Isends and Irecvs with each other process).  Each version also contains a locality-aware optimization.

### Alltoall : 
The file alltoall.c contains methods for performing the bruck alltoall algorithm and point-to-point communication (all processes perform Isends and Irecvs with each other process).  This file contains locality-aware aggregation for the p2p version, and a locality-aware bruck alltoall is in progress.

### Alltoallv : 
The file alltoallv.c contains point-to-point communication for the all-to-allv operation, and a locality-aware optimization for this.  A persistent version of the locality-aware alltoallv is in progress to improve load balancing without significant overheads.

## Neighborhood Collectives : 
The neighborhood collective operations are within the folder src/neighborhood.

### Dist Graph Create : 
To use the MPI Advance optimizations for neighborhood collectives, create the topology communicator with MPIX_Dist_graph_create_adjacent (in dist_graph.c).

### Neighbor Alltoallv : 
A standard neighbor alltoallv and locality-aware version are both implemented in neighbor.c.  To use these, call the dist graph create adjacent method above, followed by MPIX_Neighbor_alltoallv_init().

### Neighbor Alltoallv : 
A standard neighbor alltoallw version is implemented in neighbor.c.  To use this, call the dist graph create adjacent method above, followed by MPIX_Neighbor_alltoallw_init().