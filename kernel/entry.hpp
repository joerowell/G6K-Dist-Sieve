#ifndef INCLUDED_ENTRY_HPP
#define INCLUDED_ENTRY_HPP

#include "constants.hpp"
#include <array>

/**
    struct Entry:

    A vector with two representations, and side information.
    This is used to store the actual points of our main database db.

    The "key" entry to Entries is x, which stores the coordinate wrt to the
basis B. All other data are computed from x. The other data are guaranteed to be
consistent with x (and the sieve context defining the GSO) except during a
change of context.

    When modifying points, there is the recompute_data_for_entry member function
template of Siever to restore the invariants. Also, we require that the sizes of
x and yr are always identical and identical to n. Since we use a fixed-length
array, this means that only the first n entries are meaningful.
**/

struct Entry {
  std::array<ZT, MAX_SIEVING_DIM> x; // Vector coordinates in local basis B.
  std::array<LFT, MAX_SIEVING_DIM>
      yr; // Vector coordinates in gso basis renormalized by the rr[i] (for
          // faster inner product)
  CompressedVector c; // Compressed vector (i.e. a simhash)
  UidType uid; // Unique identifier for collision detection (essentially a hash)
  FT len = 0.; // (squared) length of the vector, renormalized by the local
               // gaussian heuristic
  std::array<LFT, OTF_LIFT_HELPER_DIM>
      otf_helper; // auxiliary information to accelerate otf lifting of pairs
};
#endif
