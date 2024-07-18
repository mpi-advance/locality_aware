#include "timer.hpp"

template <typedef F>
void time_function(F* f, MPI_Comm comm)
{
    double t0, tfinal;
    int n_iter;

    // Warm-Up
    f();

    // Calculate Iteration Count
    MPI_Barrier(comm);
    t0 = MPI_Wtime();
    f();
    tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (t0 > 1)
        n_iter = 1;
    else
        n_iter = (int)(1.0 / t0);


    // Accurate Timing
    MPI_Barrier(comm);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        f();
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, comm);
    return t0;
}

#endif
