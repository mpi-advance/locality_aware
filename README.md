# Overview
This repository performs locality-aware optimizations for standard MPI collectives as well as neighborhood collectives.


# Building Instructions

## Prequistites and Dependencies
cmake 3.21
C 99, C++ 11
MPI
HIP or CUDA for heterogeneous functions. 

## Building
```
mkdir build
cd build
cmake .. <build options>
```
### Build options
- -DBENCHMARKS (Default on) <br>
	This option enables the building of the benchmarks in the top level of the repo. The benchmark executable are built and available for use. Input files for the benchmarks can be found in /Test_data 

- -DENABLE_UNIT_TESTS (Default on) <br>
	This option enables ctest support for quick testing of proper functionality of the library. Tests can be run by either `make test` or by running `ctest` in the build directory.

- -DUSE_CUDA (Default off) <br>
	Build the library with cuda support. In order to use this option you will additionally need to provide the target Nvidia architecture. 
	The cuda architecture may already be set in the environment. If not it can be found by searching for the gpu model on Nvidia's [compute capability list](https://developer.nvidia.com/cuda-gpus) and supplying the compute capability as 
-DCMAKE_CUDA_ARCHITECTURES. You should ignore the decimal when supplying the architecture. For example: a GeForce RTX 4080 has a compute capability of 8.9 thus to build for it you would use `-DCMAKE_CUDA_ARCHITECTURES=89`.

- -DUSE_CUDA (Default off) <br>
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
  
### Alltoall
```c
int MPIL_Alltoall(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* comm);
```
#### Set function
- int MPIL_Set_alltoall_algorithm(enum AlltoallMethod algorithm);

#### Implemented Algorithm Combinations
    ALLTOALL_GPU_PAIRWISE,
    ALLTOALL_GPU_NONBLOCKING,
    ALLTOALL_CTC_PAIRWISE,
    ALLTOALL_CTC_NONBLOCKING,
    ALLTOALL_PAIRWISE,
    ALLTOALL_NONBLOCKING,
    ALLTOALL_HIERARCHICAL_PAIRWISE,
    ALLTOALL_HIERARCHICAL_NONBLOCKING,
    ALLTOALL_MULTILEADER_PAIRWISE,
    ALLTOALL_MULTILEADER_NONBLOCKING,
    ALLTOALL_NODE_AWARE_PAIRWISE,
    ALLTOALL_NODE_AWARE_NONBLOCKING,
    ALLTOALL_LOCALITY_AWARE_PAIRWISE,
    ALLTOALL_LOCALITY_AWARE_NONBLOCKING,
    ALLTOALL_MULTILEADER_LOCALITY_PAIRWISE,
    ALLTOALL_MULTILEADER_LOCALITY_NONBLOCKING,
    ALLTOALL_PMPI
	
### Alltoallv
```c
int MPIL_Alltoallv(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm);
				   ```
#### Set functions
- int MPIL_Set_alltoallv_algorithm(enum AlltoallvMethod algorithm);
#### Implemented Algorithm Combinations
    ALLTOALLV_GPU_PAIRWISE,
    ALLTOALLV_GPU_NONBLOCKING,
    ALLTOALLV_CTC_PAIRWISE,
    ALLTOALLV_CTC_NONBLOCKING,
    ALLTOALLV_PAIRWISE,
    ALLTOALLV_NONBLOCKING,
    ALLTOALLV_BATCH,
    ALLTOALLV_BATCH_ASYNC,
    ALLTOALLV_PMPI

### CRS Collective API
Optimized algorithms for CRS operations. 

```c
int MPIL_Info_init(MPIL_Info** info);
int MPIL_Info_free(MPIL_Info** info);
int MPIL_Alltoall_crs(const int send_nnz,
                      const int* dest,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      const void* sendvals,
                      int* recv_nnz,
                      int** src_ptr,
                      int recvcount,
                      MPI_Datatype recvtype,
                      void** recvvals_ptr,
                      MPIL_Info* xinfo,
                      MPIL_Comm* xcomm);
int MPIL_Alltoallv_crs(const int send_nnz,
                       const int send_size,
                       const int* dest,
                       const int* sendcounts,
                       const int* sdispls,
                       MPI_Datatype sendtype,
                       const void* sendvals,
                       int* recv_nnz,
                       int* recv_size,
                       int** src_ptr,
                       int** recvcounts_ptr,
                       int** rdispls_ptr,
                       MPI_Datatype recvtype,
                       void** recvvals_ptr,
                       MPIL_Info* xinfo,
                       MPIL_Comm* comm);
int MPIL_Info_init(MPIL_Info** info);
int MPIL_Info_free(MPIL_Info** info);
```
#### Set functions
- int MPIL_Set_alltoall_crs(enum AlltoallCRSMethod algorithm);
- int MPIL_Set_alltoallv_crs(enum AlltoallvCRSMethod algorithm);

#### Implemented Algorithm Combinations
    ALLTOALL_CRS_RMA,
    ALLTOALL_CRS_NONBLOCKING,
    ALLTOALL_CRS_NONBLOCKING_LOC,
    ALLTOALL_CRS_PERSONALIZED,
    ALLTOALL_CRS_PERSONALIZED_LOC
	ALLTOALLV_CRS_NONBLOCKING,
    ALLTOALLV_CRS_NONBLOCKING_LOC,
    ALLTOALLV_CRS_PERSONALIZED,
    ALLTOALLV_CRS_PERSONALIZED_LOC

#### Call order
- MPIL_Info_init();
- MPIL_Set_alltoall_crs()
- MPIL_Alltoall_crs()




### Neighborhood Collectives
For neighborhood collectives you need to create a Neighborhood communicator this can be done by using
- MPIL_Dist_graph_create_adjacent. Once you have a Neighborhood comm, you can store that configuration in a MPIL_Topo object using 
- MPI_Topo_from_neighbor_comm. Once the MPIL_Topo is created you can use it directly with the 'topo' API calls. 

```c 
int MPIL_Dist_graph_create_adjacent(MPI_Comm comm_old,
                                    int indegree,
                                    const int sources[],
                                    const int sourceweights[],
                                    int outdegree,
                                    const int destinations[],
                                    const int destweights[],
                                    MPIL_Info* info,
                                    int reorder,
                                    MPIL_Comm** comm_dist_graph_ptr);
int MPIL_Topo_from_neighbor_comm(MPIL_Comm* comm, MPIL_Topo** mpil_topo_ptr);

int MPIL_Neighbor_alltoallv(const void* sendbuf,
                            const int sendcounts[],
                            const int sdispls[],
                            MPI_Datatype sendtype,
                            void* recvbuf,
                            const int recvcounts[],
                            const int rdispls[],
                            MPI_Datatype recvtype,
                            MPIL_Comm* comm);
int MPIL_Neighbor_alltoallv_topo(const void* sendbuf,
                                 const int sendcounts[],
                                 const int sdispls[],
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcounts[],
                                 const int rdispls[],
                                 MPI_Datatype recvtype,
                                 MPIL_Topo* topo,
                                 MPIL_Comm* comm);							
int MPIL_Neighbor_alltoallv_init(const void* sendbuf,
                                 const int sendcounts[],
                                 const int sdispls[],
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcounts[],
                                 const int rdispls[],
                                 MPI_Datatype recvtype,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** request_ptr);
int MPIL_Neighbor_alltoallv_init_topo(const void* sendbuf,
                                      const int sendcounts[],
                                      const int sdispls[],
                                      MPI_Datatype sendtype,
                                      void* recvbuf,
                                      const int recvcounts[],
                                      const int rdispls[],
                                      MPI_Datatype recvtype,
                                      MPIL_Topo* topo,
                                      MPIL_Comm* comm,
                                      MPIL_Info* info,
                                      MPIL_Request** request_ptr);
int MPIL_Neighbor_alltoallv_init_ext(const void* sendbuf,
                                     const int sendcounts[],
                                     const int sdispls[],
                                     const long global_sindices[],
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcounts[],
                                     const int rdispls[],
                                     const long global_rindices[],
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm,
                                     MPIL_Info* info,
                                     MPIL_Request** request_ptr);
int MPIL_Neighbor_alltoallv_init_ext_topo(const void* sendbuf,
                                          const int sendcounts[],
                                          const int sdispls[],
                                          const long global_sindices[],
                                          MPI_Datatype sendtype,
                                          void* recvbuf,
                                          const int recvcounts[],
                                          const int rdispls[],
                                          const long global_rindices[],
                                          MPI_Datatype recvtype,
                                          MPIL_Topo* topo,
                                          MPIL_Comm* comm,
                                          MPIL_Info* info,
                                          MPIL_Request** request_ptr);
```
#### Call order
- MPIL_Dist_graph_create_adjacent()
- MPIL_Set_alltoall_neighbor_alogrithm()
- MPIL_Neighbor_alltoall()



### Structs and Classes. 
The library provides the following opaque structs. Limited access and control of the interior of these structs is available through API calls. For more information see doxygen docs. 
- MPIL_Comm
- MPIL_Info
- MPIL_Topo
- MPIL_Request

For functions provided by the library the provided functions should be used in place of the standard MPI structs of the similar name. 

### Support Functions
Functions available to user to interact with structs
```c
int MPIL_Alloc(void** pointer, const int bytes);
int MPIL_Free(void* pointer);
int MPIL_Comm_init(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm);
int MPIL_Comm_free(MPIL_Comm** xcomm_ptr);
int MPIL_Comm_device_init(MPIL_Comm* xcomm);
int MPIL_Comm_device_free(MPIL_Comm* xcomm);
int MPIL_Comm_win_init(MPIL_Comm* xcomm, int bytes, int type_bytes);
int MPIL_Comm_win_free(MPIL_Comm* xcomm);
int MPIL_Comm_leader_init(MPIL_Comm* xcomm, int procs_per_leader);
int MPIL_Comm_leader_free(MPIL_Comm* xcomm);
int MPIL_Comm_req_resize(MPIL_Comm* xcomm, int n);
int MPIL_Comm_update_locality(MPIL_Comm* xcomm, int ppn);
int MPIL_Comm_tag(MPIL_Comm* comm, int* tag);
int MPIL_Comm_topo_init(MPIL_Comm* xcomm);
int MPIL_Comm_topo_free(MPIL_Comm* xcomm);
```
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
