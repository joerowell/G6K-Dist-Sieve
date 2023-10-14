#ifndef INCLUDED_DIST_SIEVER_HPP
#define INCLUDED_DIST_SIEVER_HPP

#include "mpi.h"
#include "siever.h"

/**
   DistSiever.

   \brief This object realises non-root ranks on a network. More specifically,
this class acts as a rank that doesn't have Cython running on it. All other
aspects of the siever are essentially the same.

   The structure of this class is as follows.

   1. The DistSiever contains a singular Siever object that is responsible for
most of the heavy lifting when it comes to sieving. The siever object is
responsible for actually sieving over vectors and doing other areas of work.
   2. This class borrows the MPI object from the Siever to drive most of its
life. The class essentially contains a single method (run) that is called by the
rest of the program. This runs the distributed sieve until the root rank finally
sends a "done" method when the sieve finishes. You can view this portion as an
event loop.
   3. A particular siever is updated via messages from the root. This is pretty
straightforward to achieve in practice, and all required methods exist in
      mpi_wrapper.
**/
class DistSiever {
public:
  DistSiever(SieverParams p) noexcept;
  void run() noexcept;

private:
  /**
     siever. This siever object drives sieving.
   **/
  Siever siever;
};

#endif
