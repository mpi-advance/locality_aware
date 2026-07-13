#ifndef LA_GLOBAL_COMM_HPP
#define LA_GLOBAL_COMM_HPP

#include <mpi.h>

#include <iostream>
#include <memory>
#include <vector>

namespace Communicator
{
    /** @brief Global MPI Communicator to replace MPI_COMM_WORLD inside this library */
    extern MPI_Comm WORLD_COMM;

    /** @brief A reference-counted wrapper around an MPI communicator
     * @details This class uses a shared pointer around an MPI_Comm object, which is
     * manually allocated when the main constructor is called. If any of copy/move
     * constructors are used, the shared_ptr semantics kick in an allow for reference
     * counted communicators.
     *
     * This class also provides helper casting operations so that users do not need to
     * manually cast to "MPI_Comm" or even "MPI_Comm*" when using this class. Users are
     * free to free the one passed to the constructor, but users should NOT free the
     * communicator this class holds onto. This is taken care of by the custom deleter
     * provided to the shared_ptr (see CommDeleter)
     *
     * Take care when using this struct inside another struct that is malloc'ed (see
     * ::initialize_comm_object)
     **/
    class CachedComm
    {
      public:
        /** @brief Main constructor for a CachedComm.
         * @details Allocates a new MPI_Comm on the heap for the internal shared_ptr. Uses
         * explicit to require the user to manually specify one must be created.
         **/
        explicit CachedComm(const MPI_Comm comm)
            : my_comm(new MPI_Comm(comm), CommDeleter{})
        {
        }

        /** @brief Function to automatically cast to MPI_Comm
         * @return The MPI_Comm inside the shared_ptr. If the shared_ptr has somehow been
         * cleared, MPI_COMM_NULL is returned.
         **/
        operator MPI_Comm() const
        {
            return my_comm ? *my_comm : MPI_COMM_NULL;
        }

        /** @brief Function to automatically cast to MPI_Comm*
         * @return The address of the MPI_Comm object inside the shared_ptr.
         **/
        operator MPI_Comm*() const
        {
            return my_comm.get();
        }

      private:
        /** @brief Helper class for managing the MPI_Comm objects inside CachedComm */
        struct CommDeleter
        {
            /** @brief The deleter function to be called.
             * @details If the pointer is valid, this function will delete it.
             * If the value at the pointer does not point to MPI_COMM_WORLD,
             * MPI_COMM_SELF, or MPI_COMM_NULL, then this function attempts to free it.
             * Before doing so, it checks if MPI has been finalized, and if it has, it
             * prints a warning and does not free. Otherwise, the MPI_Comm is passed to
             * MPI_Comm_free.
             **/
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

        /** @brief The reference counted MPI_Comm object. */
        std::shared_ptr<MPI_Comm> my_comm;
    };

    /** @brief A helper type to emulate a map "key" inside a vector */
    using KeyTypes = std::tuple<MPI_Group, int>;
    /** @brief The type of the cached comms "map" to be stored inside a vector */
    using MapPairType = std::pair<KeyTypes, CachedComm>;

    /** @brief "Map" of cached local (per-node) MPI communicators */
    extern std::vector<MapPairType> cached_local_comms;
    /** @brief "Map" of cached group (rank per-node) MPI communicators */
    extern std::vector<MapPairType> cached_group_comms;

    /** @brief "Map" of cached local (per-node) MPI communicators */
    extern std::vector<MapPairType> cached_leader_comms;
    /** @brief "Map" of cached group (rank per-node) MPI communicators */
    extern std::vector<MapPairType> cached_leader_group_comms;
    /** @brief "Map" of cached group (rank per-node) MPI communicators */
    extern std::vector<MapPairType> cached_leader_local_comms;

    /** @brief Destructor for the MPI Groups stores inside ::cached_local_comms and
     * ::cached_group_comms.
     **/
    void clear_comm_caches();

}  // namespace Communicator

#endif
