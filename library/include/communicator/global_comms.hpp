#ifndef LA_GLOBAL_COMM_HPP
#define LA_GLOBAL_COMM_HPP

#include <mpi.h>

#include <iostream>
#include <map>
#include <memory>

namespace Communicator
{
    /**@brief Global MPI Communicator to replace MPI_COMM_WORLD inside this library */
    extern MPI_Comm WORLD_COMM;

    class CachedComm
    {
      public:
        explicit CachedComm(MPI_Comm comm) : my_comm(new MPI_Comm(comm), CommDeleter{}) {
        }

        operator MPI_Comm() const
        {
            return my_comm ? *my_comm : MPI_COMM_NULL;
        }

        operator MPI_Comm*() const
        {
            return my_comm.get();
        }

        void test()
        {
            std::cout << my_comm.use_count() << std::endl;
        }

      private:
        struct CommDeleter
        {
            void operator()(MPI_Comm* comm) const
            {
                if (comm)
                {
                    if (*comm != MPI_COMM_WORLD && *comm != MPI_COMM_SELF &&
                        *comm != MPI_COMM_NULL)
                    {
                        int finalized = 0;
                        MPI_Finalized(&finalized);

                        if (!finalized)
                        {
                            MPI_Comm_free(comm);
                        }
                        else
                        {
                            std::cerr << "[Warning] MPI already finalized. Cannot free "
                                         "MPI_Comm safely."
                                      << std::endl;
                        }
                    }
                    delete comm;  // Delete the dynamically allocated MPI_Comm variable
                                  // itself
                }
            }
        };

        std::shared_ptr<MPI_Comm> my_comm;
    };

    extern std::map<std::tuple<MPI_Comm, int>, CachedComm> cached_local_comms;
    extern std::map<std::tuple<MPI_Comm, int>, CachedComm> cached_group_comms;

}  // namespace Communicator

#endif