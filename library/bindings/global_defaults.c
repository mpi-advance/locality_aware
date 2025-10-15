#include "locality_aware.h"

enum AlltoallMethod mpil_alltoall_implementation = ALLTOALL_PAIRWISE;

enum AlltoallvMethod mpil_alltoallv_implementation = ALLTOALLV_PAIRWISE;

enum NeighborAlltoallvMethod mpil_neighbor_alltoallv_implementation =
    NEIGHBOR_ALLTOALLV_STANDARD;
	
enum NeighborAlltoallvInitMethod mpil_neighbor_alltoallv_init_implementation =
    NEIGHBOR_ALLTOALLV_INIT_STANDARD;
	
enum AlltoallCRSMethod mpil_alltoall_crs_implementation =
    ALLTOALL_CRS_PERSONALIZED;
	
enum AlltoallvCRSMethod mpil_alltoallv_crs_implementation =
    ALLTOALLV_CRS_PERSONALIZED;


void MPIL_set_alltoall_algorithm(enum AlltoallMethod algorithm){
	mpil_alltoall_implementation = (enum AlltoallMethod)algorithm;
}
void MPIL_set_alltoallv_algorithm(enum AlltoallvMethod algorithm){
	mpil_alltoallv_implementation = (enum AlltoallvMethod)algorithm;
	
}

void MPIL_set_alltoall_neighbor_alogorithm(enum NeighborAlltoallvMethod algorithm){
	mpil_neighbor_alltoallv_implementation = (enum NeighborAlltoallvMethod)algorithm;
}
void MPIL_set_alltoallv_neighbor_init_alogorithm(enum NeighborAlltoallvInitMethod algorithm){
	mpil_neighbor_alltoallv_init_implementation = (enum NeighborAlltoallvInitMethod)algorithm;
}

void MPIL_set_alltoall_crs(enum AlltoallCRSMethod algorithm){
	mpil_alltoall_crs_implementation = (enum AlltoallCRSMethod) algorithm;
}
void MPIL_set_alltoallv_crs(enum AlltoallvCRSMethod algorithm){
	mpil_alltoallv_crs_implementation = (enum AlltoallvCRSMethod)algorithm;
}


//////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>

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
				mpil_neighbor_alltoallv_implementation = (enum NeighborAlltoallvMethod)algorithm;
				break;
			case Neighboralltoallv_init:
				mpil_neighbor_alltoallv_init_implementation = (enum NeighborAlltoallvInitMethod)algorithm;
				break;
			default:
				printf("ERROR:: UNRECOGNIZED MPIL FUNCTION\n");
				exit(EXIT_FAILURE);
		};			
	}else{
		printf("ERROR Invalid assignment of function");
	}			
}