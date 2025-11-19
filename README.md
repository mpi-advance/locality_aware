# Overview
The locality_aware repository offers locality-aware optimizations for the standard and neighborhood MPI Alltoall collective. This library also provides these optimizations for the the "v" variants of these collectives.


# Building Instructions
The locality_aware library is a straightforward CMake projects with modest dependencies.

## Software Dependencies
- CMake 3.21
- C 11
- C++ 11
- MPI
- HIP or CUDA for GPU-based optimizations, if enabled

## Building
```
mkdir build
cd build
cmake <build options> ..
```
### Build options
- `-DGPU_AWARE` (Default `ON`) <br>
    Allow the library to build with supplied GPU type if available (see USE_CUDA and USE_HIP) 
- `-DUSE_CUDA` (Default `OFF`) <br>
	Build the library with CUDA support. In order to use this option you may need to provide the target GPU architecture if it is not set in the environment. The specific architecture of an NVIDIA GPU can be found by searching for the GPU model on NVIDIA's [compute capability list](https://developer.nvidia.com/cuda-gpus) and supplying the compute capability via `-DCMAKE_CUDA_ARCHITECTURES`. You should ignore the decimal when supplying the architecture. For example: a GeForce RTX 4080 has a compute capability of 8.9 thus to build for it you would use `-DCMAKE_CUDA_ARCHITECTURES=89`.
- `-DUSE_HIP` (Default `OFF`) <br>
	Build the library with HIP support. In order to use this option you may need to provide the target GPU architecture if it is not set in the environment. The specific architecture of an AMD GPU can be found by searching for the GPU model at  [ROCM capability list](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) and supplying the compute capability via `-DCMAKE_HIP_ARCHITECTURES`. For example: `-DCMAKE_HIP_ARCHITECTURES=gfx942`
- `-DBENCHMARKS` (Default `ON`) <br>
	This option enables the building of the benchmarks in the top level of the repo. The benchmark executable are built and available for use. Input files for the benchmarks can be found in `<top repository level>/test_data` 
- `-DENABLE_UNIT_TESTS` (Default `ON`) <br>
	This option enables ctest support for quick testing of proper functionality of the library. Tests can be run by either `make test` or by running `ctest` in the build directory.
- `-DTEST_PROCS`(Default `16`) <br>
	Controls the number of processes used during ctests.
- `-DMPIRUN` (Default `mpirun`) <br>
	Set the command to used to run ctests.  


# Using the Library

All user-facing structs and APIs can be found in repository file: `include/locality_aware.h`. 
The produced library file will be called `liblocality_aware.<a/so>`. This project will also install a CMake module file to enable CMake to find the locality_aware library for use in other projects.

## APIs
This library offers multiple algorithms to modify the behavior of MPI_Alltoall and MPIL_Alltoallv collective operations. A majority of these functions share a common interface with the official functions in the MPI standard and simply utilize our MPIL prefix, whiles require a few more inputs and/or MPIL-specific objects. The latter APIs do not have an existing entry in the MPI standard to describe the input parameters, so please see internal documentation for function details.  

Each of the MPIL collective operations contain a few options of algorithms to run internally. The algorithm used is defined in an variable external to the function and can be changed at the user level to swap the algorithm being run underneath. The set of available algorithm options available for each call, and how to select them, are shown below. 

### Selecting an algorithm
A majority of the API calls are wrappers that invoke internal functions to complete the operation. The API calls follow the same parameter setup as the associated MPI function. 
Each wrapper looks for a set global variable to determine which algorithm to use. Users can set these global variables using the `MPIL_Set_<>` functions, after which all calls of the associated alltoall operation will use the new algorithm. If no algorithm is selected or an invalid enum is provided, each function uses the default algorithm noted below. 
  
### Alltoall
```c
int MPIL_Alltoall(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* comm);

int MPIL_Set_alltoall_algorithm(enum AlltoallMethod algorithm);  

enum AlltoallMethod{
    ALLTOALL_GPU_PAIRWISE,                  // Default
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
};		  
```
	
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

int MPIL_Set_alltoallv_algorithm(enum AlltoallvMethod algorithm)

enum AlltoallvMethod{
    ALLTOALLV_GPU_PAIRWISE,    // Default
    ALLTOALLV_GPU_NONBLOCKING,
    ALLTOALLV_CTC_PAIRWISE,
    ALLTOALLV_CTC_NONBLOCKING,
    ALLTOALLV_PAIRWISE,
    ALLTOALLV_NONBLOCKING,
    ALLTOALLV_BATCH,
    ALLTOALLV_BATCH_ASYNC,
    ALLTOALLV_PMPI
}
```

### Compressed-row Storage Alltoall Variants

The locality_aware library also offers an alltoall and alltoallv variant that has been modified to work specifically with compressed row storage layout. These calls do require additional parameters than the normal alltoall call (see the header file for more details).

```c
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

int MPIL_Set_alltoall_crs(enum AlltoallCRSMethod algorithm);
			
enum AlltoallCRSMethod{
    ALLTOALL_CRS_RMA,
    ALLTOALL_CRS_NONBLOCKING,
    ALLTOALL_CRS_NONBLOCKING_LOC,
    ALLTOALL_CRS_PERSONALIZED,      // Default
    ALLTOALL_CRS_PERSONALIZED_LOC			
}

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
					   
int MPIL_Set_alltoallv_crs(enum AlltoallvCRSMethod algorithm);

enum AlltoallvCRSMethod{
	ALLTOALLV_CRS_NONBLOCKING,
    ALLTOALLV_CRS_NONBLOCKING_LOC,
    ALLTOALLV_CRS_PERSONALIZED,     // Default
    ALLTOALLV_CRS_PERSONALIZED_LOC

}
```

### Neighborhood Collectives 
For neighborhood collectives, you first need to create a neighborhood communicator before the opration. This can be done by using `MPIL_Dist_graph_create_adjacent`, in the same way as its MPI counterpart. In addition to `MPIL_Neighbor_alltoallv`, the locality_aware library offers a variant on the neighbor alltoall that takes a normal (MPIL) communicator and an additional object that specifies the topology. This object (`MPIL_Topo`) allows for a first-class topology object that is not tied to a communicator, and thus could be easily adjusted or regenrated without needing a new communicator. If you have already created a neighborhood communicator, you can use `MPIL_Topo_from_neighbor_comm` to copy the topology object inside the communicator.

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

int MPIL_Set_alltoallv_neighbor_alogorithm(enum NeighborAlltoallvMethod algorithm);

enum NeighborAlltoallvMethod
{
    NEIGHBOR_ALLTOALLV_STANDARD,  // Default
    NEIGHBOR_ALLTOALLV_LOCALITY
};
```
#### Persistent Neighborhood Alltoallv
There are also persistent versions of the nieghborhood alltoallv collectives:

```c								 
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

int MPIL_Set_alltoallv_neighbor_init_alogorithm(enum NeighborAlltoallvInitMethod algorithm);

enum NeighborAlltoallvInitMethod
{
    NEIGHBOR_ALLTOALLV_INIT_STANDARD, // DEFAULT
    NEIGHBOR_ALLTOALLV_INIT_LOCALITY
};
```

### Structs and Classes. 
The library provides the following opaque structs. Limited access and control of the interior of these structs is available through API calls. For more information, please see the doxygen documentation. 
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
int MPIL_Info_init(MPIL_Info** info);
int MPIL_Info_free(MPIL_Info** info);
```
# Repository Layout
The locality_aware repository is laid out as follows:
- *Include* 
- *Benchmarks* Contains source code for several benchmarks used in publications.
- *Test Data* Contains input files for testing of Benchmarks. 
- *Library* Contains the source code for the library. It is divided into four parts: 
	- **bindings:** contains the implementations of all the user facing functions. These commonly call functions deeper inside the library. 
	- **include:** contains any internal headers necessary for building.
	- **source:** contains implementations of internal functions used by the bindings.
	- **test:**   contains functions and code for conducting the unit tests. 


# Acknowledgements
This work has been funded by the National Science Foundation through grants 2514054, 2450093, 2412182, CNS2450092, CCF2338077, CCF2151022. This work was also performed with support from the U.S. Department of Energy's National Nuclear Security Administration (NNSA) under the Predictive Science Academic Alliance Program (PSAAP-III), Award DE-NA0003966. 

Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation or the U.S. Department of Energy's National Nuclear Security Administration.
