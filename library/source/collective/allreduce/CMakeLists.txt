target_sources(locality_aware PRIVATE 
    allreduce_recursive_doubling.cpp
    allreduce_dissemination_loc.cpp
    allreduce_dissemination_radix.cpp
    allreduce_pmpi.cpp
)
