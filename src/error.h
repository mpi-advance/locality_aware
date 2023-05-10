#pragma once

#include <mpi.h>

#define MPI_ADVANCE_SUCCESS_OR_RETURN(_code) \
  {if (MPI_SUCCESS != _code) return _code; }
