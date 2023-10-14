#ifndef INCLUDED_MPI_CAST_HPP
#define INCLUDED_MPI_CAST_HPP

#include "mpi.h"   // Needed for MPI things.
#include <cstdint> // Needed for uint64 casts.

/**
   MPI_Cast. This namespace exists solely to highlight a discrepancy across
   MPI distributions. Briefly, in siever.h we allow the siever to accept an
MPI_Comm as a 64-bit integer (this is to stop G6K from needing to always import
MPI).

   The problem is that across different MPI distributions the type of MPI_Comm
changes. This means that in some situations only one of these is legal:

   \code{.cpp}
      // Assume the comm is called comm
      auto as_uint64_s = static_cast<uint64_t>(comm);
      auto as_uint64_r = reinterpret_cast<uint64_t>(comm);
   \endcode{}

   This is because on some platforms MPI_Comm is an integral type (and thus the
static_cast is the only legal one), whereas on others MPI_Comm is a pointer type
(and thus the reinterpret_cast is the only legal one).

To fix this problem, this namespace provides two functions that deal with this
for us. At the moment these just use C style casts, but hiding them in this way
allows us to abstract away this detail.
**/

namespace MPI_Cast {

// Note: this static assertion is here to make sure that we aren't
// deploying this on systems where we'd be better suited to use a
// uintptr_t: whilst semantically that's clearer, it's also semantically
// incorrect if MPI_Comm is not a pointer type.
static_assert(sizeof(uintptr_t) == sizeof(uint64_t), "Error: ");

/**
   uint64_t_to_mpi_comm. This function accepts a uint64_t `in` as input
   and casts it to an MPI_Comm, returning the result. This function is not
 guaranteed to return a value that refers to a valid MPI_Comm unless the `in`
 value was created using mpi_comm_to_uint64. This function does not throw.
   @param[in] in: the value to cast.
   @return `in` cast to a MPI_Comm.
 **/
inline MPI_Comm uint64_to_mpi_comm(const uint64_t in) noexcept;

/**
   mpi_comm_to_uint64. This function accepts an MPI_Comm `in` as input
   and casts it to a uint64_t, returning the result. This function must be
supplied with a valid MPI_Comm as `in` for later reconstruction. This function
does not throw.
   @param[in] in: the MPI_Comm to cast.
   @return `in` cast to a uint64_t.
**/
inline uint64_t mpi_comm_to_uint64(const MPI_Comm in) noexcept;

} // namespace MPI_Cast

// Inline definitions go here.
#include "mpi_cast.inl"
#endif
