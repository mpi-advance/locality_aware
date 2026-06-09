#include "communicator/global_comms.hpp"

MPI_Comm Communicator::WORLD_COMM = MPI_COMM_NULL;
std::map<std::tuple<MPI_Comm, int>, Communicator::CachedComm> Communicator::cached_local_comms;
std::map<std::tuple<MPI_Comm, int>, Communicator::CachedComm> Communicator::cached_group_comms;
