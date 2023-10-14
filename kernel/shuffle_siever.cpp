#ifdef G6K_MPI
#include "shuffle_siever.hpp"
#ifndef DOCTEST_CONFIG_DISABLE
#include "doctest/extensions/doctest_mpi.h"
#endif

ShuffleSiever::ShuffleSiever(MPI_Comm &comm_)
    : Topology{}, right{}, left{}, left_uid_table{},
      right_uid_table{}, comm{comm_} {

  /*
  // The ring is of height 1.
  constexpr auto n_y = 1;
  // Work out how wide the ring is.
  int n_x;
  MPI_Comm_size(comm, &n_x);

  int ndims[2]{n_x, n_y};

  // This essentially means: the ring wraps around at both ends.
  constexpr int period[]{true, true};
  // We also allow MPI to work out how to put all of this together.
  // It's an int for legacy reasons.
  constexpr int reorder = true;
  MPI_Comm cart_comm;
  MPI_Cart_create(comm, 2, ndims, period, reorder, &cart_comm);

  // This says: we are retrieving the node to the left of us
  // and the node to the right.
  MPI_Cart_shift(cart_comm, 1, 0, &left, &right);

  // Write out the comm. The parent will still need to set this explicitly to
  // set up the network.
  this->comm = cart_comm;
  */
}

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("get_type", 1) {
  ShuffleSiever s(test_comm);
  CHECK(s.get_type() == DistSieverType::ShuffleSiever);
}
#endif

DistSieverType ShuffleSiever::get_type() const noexcept {
  return DistSieverType::ShuffleSiever;
}

void ShuffleSiever::setup(const bool) noexcept {}
void ShuffleSiever::send_entries_via_cdb(const std::vector<CompressedEntry> &,
                                         const std::vector<Entry> &,
                                         const unsigned,
                                         const unsigned) noexcept {}

void ShuffleSiever::receive_entries_via_cdb(std::vector<CompressedEntry> &,
                                            std::vector<Entry> &,
                                            const unsigned, const unsigned) {}

void ShuffleSiever::receive_entries(std::vector<Entry> &, const unsigned,
                                    const unsigned) noexcept {}

void ShuffleSiever::send_entries(const std::vector<Entry> &, const unsigned,
                                 const unsigned) noexcept {}
#endif
