#ifndef BENCHMARK_TIMER_HPP
#define BENCHMARK_TIMER_HPP

#include "mpi_advance.hpp"

template <typedef F>
void time_function(F* f, MPI_Comm comm);

#endif
