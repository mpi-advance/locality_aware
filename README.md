# Overview
This repository performs locality-aware optimizations for standard MPI collectives as well as neighborhood collectives.


# Building Instructions

## Prequistites and Dependencies
cmake 3.21
C 99, C++ 11
MPI
HIP or CUDA for heterogeneous functions. 

## Building
mkdir build
cd build
cmake .. <build options>

### Build options
- -DBENCHMARKS <br>
	This option enables the building of the benchmarks in the top level of the repo. The benchmark executable are built and available for use. Input files for the benchmarks can be found in /Test_data 

- -DENABLE_UNIT_TESTS <br>
	This option enables ctest support for quick testing of proper functionality of the library. Tests can be run by either `make test` or by running `ctest` in the build directory.

- -DUSE_CUDA <br>
	Build the library with cuda support. In order to use this option you will additionally need to provide the target Nvidia architecture. 
	The cuda architecture may already be set in the environment. If not it can be found by searching for the gpu model on Nvidia's [compute capability list](https://developer.nvidia.com/cuda-gpus) and supplying the compute capability as 
-DCMAKE_CUDA_ARCHITECTURES. You should ignore the decimal when supplying the architecture. For example: a GeForce RTX 4080 has a compute capability of 8.9 thus to build for it you would use `-DCMAKE_CUDA_ARCHITECTURES=89`.

- -DUSE_HIP <br>
	Build the library with HIP support. In order to use this option you will additionally need to provide the target gpu architecture. 
	The cuda architecture may already be set in the environment. If not it can be found by searching for the gpu model at  [ROCM capability list](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) and supplying the compute capability as 

- -DCMAKE_HIP_ARCHITECTURES. For example: -DCMAKE_HIP_ARCHITECTURES=gfx942


# Using the Library

## Linking
All publicly available structs and API are included in include/locality.h. 
The produced shared library is locality_aware.so. 
It can be found by cmake by including the locality_aware package if the library is installed.   
## API

### Selecting an algorithm
A majority of the API calls are wrappers that invoke internal functions to complete the operation. The API calls follow the same parameter setup as the associated MPI function. 
Each wrapper looks for a set global variable, by setting these global variables different algorithms can be used. 
There are several different possible algorithms possible for each API call. 
for a full list of supported algorithms, look at the enums at the top of locality_aware.h
  
### Basic Collectives
- MPIL_Alltoall
- MPIL_Alltoallv
#### Set functions
- int MPIL_Set_alltoall_algorithm(enum AlltoallMethod algorithm);
- int MPIL_Set_alltoallv_algorithm(enum AlltoallvMethod algorithm);

### CRS Collectives
- MPIL_Alltoall_crs
- MPIL_Alltoallv_crs
- MPIL_Info_init
- MPIL_Info_free
#### Set functions
- int MPIL_Set_alltoall_crs(enum AlltoallCRSMethod algorithm);
- int MPIL_Set_alltoallv_crs(enum AlltoallvCRSMethod algorithm);

### Neighborhood Collectives
For neighborhood collectives you need to create a Neighborhood communicator this can be done by using
- MPIL_Dist_graph_create_adjacent. Once you have a Neighborhood comm, you can store that configuration in a MPIL_Topo object using 
- MPI_Topo_from_neighbor_comm. Once the MPIL_Topo is created you can use it directly with the 'topo' API calls. 
 
- MPIL_Dist_graph_create_adjacent
- MPIL_Topo_from_neighbor_comm
- MPIL_Nieghbor_alltoall
- MPIL_Neighbor_alltoallv_init
- MPIL_Neighbor_alltoallv_init_topo
- MPIL_Neighbor_alltoallv_init_ext
- MPIL_Neighbor_alltoallv_init_ext_topo

#### Call order
- MPIL_Dist_graph_create_adjacent()
- MPIL_Set_alltoall_neighbor_alogrithm()
- MPIL_Neighbor_alltoall()

### Modified MPI Functions
- MPIL_Alloc
- MPIL_Free
- MPIL_Comm_device_init
- MPIL_Comm_device_free
- MPIL_Comm_win_init
- MPIL_Comm_win_free
- MPIL_Comm_leader_init
- MPIL_Comm_leader_free

### Support Functions
- MPIL_Comm_init
- MPIL_Comm_free
- MPIL_Comm_req_resize
- MPIL_Comm_update_locality
- MPIL_Comm_tag
- MPIL_Topo_Free
- MPIL_Topo_Init

### Structs and Classes. 
The library provides the following opaque structs. Limited access and control of the interior of these structs is available through API calls. For more information see doxygen docs. 
- MPIL_Comm
- MPIL_Info
- MPIL_Topo
- MPIL_Request
For functions provided by the library the provided functions should be used in place of the standard MPI structs of the similar name. 

# Repository Layout
The library is split into four main parts. 
- library/bindings contains the implementations of all the user facing functions. These commonly call functions deeper inside the library. 
- library/include contains any internal headers necessary for building.
- library/src contains implementations of internal functions. 
- library/test only contains functions and code for conducting the unit tests. 

include and src are further divided into sub-folders depending on the portion of the library being implemented. 


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
To use the MPI Advance optimizations for neighborhood collectives, create the topology communicator with MPIL_Dist_graph_create_adjacent (in dist_graph.c).

### Neighbor Alltoallv : 
A standard neighbor alltoallv and locality-aware version are both implemented in neighbor.c.  To use these, call the dist graph create adjacent method above, followed by MPIX_Neighbor_alltoallv_init().

### Neighbor Alltoallv : 
A standard neighbor alltoallw version is implemented in neighbor.c.  To use this, call the dist graph create adjacent method above, followed by MPIX_Neighbor_alltoallw_init().