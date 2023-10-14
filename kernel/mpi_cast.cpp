#include "mpi_cast.hpp"
#ifndef DOCTEST_CONFIG_DISABLE
#include "doctest/extensions/doctest_mpi.h"
#endif

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("can cast between uint64_t and MPI_Comm", 1) {
  const auto as_uint64 = MPI_Cast::mpi_comm_to_uint64(test_comm);
  CHECK(MPI_Cast::uint64_to_mpi_comm(as_uint64) == test_comm);
}
#endif
