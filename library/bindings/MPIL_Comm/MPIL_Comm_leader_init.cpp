#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

int MPIL_Comm_leader_init(MPIL_Comm* xcomm, int procs_per_leader)
{
    if (xcomm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(xcomm);
    }

    int rank, local_rank;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_rank(xcomm->local_comm, &local_rank);

    // Get the group, since we will compare those
    MPI_Group global_group;
    MPI_Comm_group(xcomm->global_comm, &global_group);

    /* Lambda for searching for if a particular group/ppn combo has been used before. */
    auto search_function = [&global_group,
                            procs_per_leader](const Communicator::MapPairType& mpt) {
        if (std::get<1>(mpt.first) != procs_per_leader)
        {
            return false;
        }

        int result;
        MPI_Group_compare(global_group, std::get<0>(mpt.first), &result);
        /* Currently only care about MPI_IDENT */
        return (result == MPI_IDENT);
    };

    auto leader_comm_iter = std::find_if(Communicator::cached_leader_comms.begin(),
                                    Communicator::cached_leader_comms.end(),
                                    search_function);
    if (Communicator::cached_leader_comms.end() != leader_comm_iter)
    { /* Used cached entry, so we can free the MPI_Group */
        MPI_Group_free(&global_group);
        xcomm->leader_comm = leader_comm_iter->second;
    }
    else
    {
        MPI_Comm_split(xcomm->local_comm, local_rank / procs_per_leader, rank, (xcomm->leader_comm));
        /* Cache new local communicator into map for reuse */
        Communicator::cached_leader_comms.push_back(
            {{global_group, procs_per_leader}, xcomm->leader_comm});
    }
    int leader_rank;
    MPI_Comm_rank(xcomm->leader_comm, &leader_rank);


    MPI_Comm_group(xcomm->global_comm, &global_group);    
    auto leader_group_comm_iter = std::find_if(Communicator::cached_leader_group_comms.begin(),
                                    Communicator::cached_leader_group_comms.end(),
                                    search_function);
    if (Communicator::cached_leader_group_comms.end() != leader_group_comm_iter)
    { /* Used cached entry, so we can free the MPI_Group */
        MPI_Group_free(&global_group);
        xcomm->leader_group_comm = leader_group_comm_iter->second;
    }
    else
    {
        MPI_Comm_split(xcomm->global_comm, leader_rank, rank, (xcomm->leader_group_comm));
        /* Cache new local communicator into map for reuse */
        Communicator::cached_leader_group_comms.push_back(
            {{global_group, procs_per_leader}, xcomm->leader_group_comm});
    }



    MPI_Comm_group(xcomm->global_comm, &global_group);    
    auto leader_local_comm_iter = std::find_if(Communicator::cached_leader_local_comms.begin(),
                                    Communicator::cached_leader_local_comms.end(),
                                    search_function);
    if (Communicator::cached_leader_local_comms.end() != leader_local_comm_iter)
    { /* Used cached entry, so we can free the MPI_Group */
        MPI_Group_free(&global_group);
        xcomm->leader_local_comm = leader_local_comm_iter->second;
    }
    else
    {
        MPI_Comm_split(xcomm->local_comm, leader_rank, rank, (xcomm->leader_local_comm));
        /* Cache new local communicator into map for reuse */
        Communicator::cached_leader_local_comms.push_back(
            {{global_group, procs_per_leader}, xcomm->leader_local_comm});
    }


    return MPI_SUCCESS;
}
