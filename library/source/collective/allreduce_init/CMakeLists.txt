target_sources(locality_aware PRIVATE 
    allreduce_init_recursive_doubling.cpp
    allreduce_init_dissemination_loc.cpp
    allreduce_init_dissemination_radix.cpp
    allreduce_init_pmpi.cpp
)
