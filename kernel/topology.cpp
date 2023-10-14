#ifdef G6K_MPI
#include "topology.hpp"

// N.B We store import everything in this part, so that we don't have circular
// dependencies for the siever's types.
#include "shuffle_siever.hpp"
#include <memory>

Topology::~Topology() {}

std::unique_ptr<Topology>
Topology::build_topology(MPI_Comm &comm, const DistSieverType) noexcept {
  return std::unique_ptr<Topology>(new ShuffleSiever(comm));
}
#endif
