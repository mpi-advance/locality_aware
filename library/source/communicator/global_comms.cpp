#include "communicator/global_comms.hpp"

MPI_Comm Communicator::WORLD_COMM = MPI_COMM_NULL;
std::vector<Communicator::MapPairType> Communicator::cached_local_comms;
std::vector<Communicator::MapPairType> Communicator::cached_group_comms;

std::vector<Communicator::MapPairType> Communicator::cached_leader_comms;
std::vector<Communicator::MapPairType> Communicator::cached_leader_group_comms;
std::vector<Communicator::MapPairType> Communicator::cached_leader_local_comms;

void Communicator::clear_comm_caches()
{
    for (MapPairType& mpt : cached_local_comms)
    {
        MPI_Group_free(&(std::get<0>(mpt.first)));
    }
    cached_local_comms.clear();

    for (MapPairType& mpt : cached_group_comms)
    {
        MPI_Group_free(&(std::get<0>(mpt.first)));
    }
    cached_group_comms.clear();


    for (MapPairType& mpt : cached_leader_comms)
    {
        MPI_Group_free(&(std::get<0>(mpt.first)));
    }
    cached_leader_comms.clear();

    for (MapPairType& mpt : cached_leader_group_comms)
    {
        MPI_Group_free(&(std::get<0>(mpt.first)));
    }
    cached_leader_group_comms.clear();

    for (MapPairType& mpt : cached_leader_local_comms)
    {
        MPI_Group_free(&(std::get<0>(mpt.first)));
    }
    cached_leader_local_comms.clear();

}
