target_sources(locality_aware PRIVATE 
    allreduce_recursive_doubling.c
    allreduce_dissemination_loc.c
    allreduce_dissemination_radix.c
    allreduce_pmpi.c
)
