#include "locality_aware.h"
#include "communicator/MPIL_Comm.h"
#include <stdio.h>
#include <stdlib.h>

enum AlltoallMethod mpil_alltoall_implementation = ALLTOALL_PAIRWISE;

enum AlltoallvMethod mpil_alltoallv_implementation = ALLTOALLV_PAIRWISE;

enum NeighborAlltoallvMethod mpix_neighbor_alltoallv_implementation =
    NEIGHBOR_ALLTOALLV_STANDARD;

enum NeighborAlltoallvInitMethod mpix_neighbor_alltoallv_init_implementation =
    NEIGHBOR_ALLTOALLV_INIT_STANDARD;

void MPIL_set_alltoall_algorithm(enum AlltoallMethod algorithm){
	mpil_alltoall_implementation = (enum AlltoallMethod)algorithm;
}
void MPIL_set_alltoallv_algorithm(enum AlltoallvMethod algorithm){
	mpil_alltoallv_implementation = (enum AlltoallvMethod)algorithm;
	
}
void MPIL_set_alltoall_neighbor_alogorithm(enum NeighborAlltoallvMethod algorithm){
	mpix_neighbor_alltoallv_implementation = (enum NeighborAlltoallvMethod)algorithm;
}
void MPIL_set_alltoallv_neighbor_init_alogorithm(enum NeighborAlltoallvInitMethod algorithm){
	mpix_neighbor_alltoallv_init_implementation = (enum NeighborAlltoallvInitMethod)algorithm;
}

enum MPIL_function{
	Alltoall = 13,
	Alltoallv = 5,
	Neighboralltoallv = 2,
	Neighboralltoallv_init = 3
};

//Return 0 if algorthm selected is invalid option for supplied function. 
int validate_enum(enum MPIL_function function, int algorithm){
	if(algorithm < 0 || algorithm >= function){
		return 0;
	}
	return 1;
}

void MPIL_set_algorithm(enum MPIL_function function, int algorithm){
	validate_enum(function, algorithm);
	if(validate_enum(function, algorithm))
	{
		switch(function)
		{
			case Alltoall:
				mpil_alltoall_implementation = (enum AlltoallMethod)algorithm;
				break;
			case Alltoallv:
				mpil_alltoallv_implementation = (enum AlltoallvMethod)algorithm;
				break;
			case Neighboralltoallv:
				mpix_neighbor_alltoallv_implementation = (enum NeighborAlltoallvMethod)algorithm;
				break;
			case Neighboralltoallv_init:
				mpix_neighbor_alltoallv_init_implementation = (enum NeighborAlltoallvInitMethod)algorithm;
				break;
			default:
				printf("ERROR:: UNRECOGNIZED MPIL FUNCTION\n");
				exit(EXIT_FAILURE);
		};			
	}else{
		printf("ERROR Invalid assignment of function");
	}			
}