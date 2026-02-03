#include "communicator/global_comms.hpp"

MPI_Comm Communicator::WORLD_COMM = MPI_COMM_NULL;
MPI_Comm Communicator::NODE_COMM = MPI_COMM_NULL;

