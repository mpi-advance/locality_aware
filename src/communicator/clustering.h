#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "mpix_comm.h"


double* network_discovery(MPIX_Comm* xcomm, int size, int tag, int num_iterations); // This is only exposed temporarily for some scaling studies


#endif // CLUSTERING_H