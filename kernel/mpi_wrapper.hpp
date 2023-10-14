#ifndef INCLUDED_MPI_WRAPPER_HPP
#define INCLUDED_MPI_WRAPPER_HPP

#include "bucket_pair.hpp"
#include "constants.hpp"

#include "context_change.hpp"
#include "mpi.h"
#include "mpi_bandwidth_tracker.hpp"
#include "siever_types.hpp"
#include "test_utils.hpp"

#include <array>
#include <cstdint>
#include <ctime>
#include <limits>
#include <vector>

// Note: these forward declarations should match Siever.h. This means that if
// the type is declared as a class in Siever.h, it should also be a class here.
// This is to prevent compiler weirdness (GCC warns on this).
// All of these are needed for various function signatures.
class Siever;
class SieverParams;
class SieveStatistics;
class UidHashTable;

struct LiftEntry;
struct Entry;
struct CompressedEntry;

#ifdef MPI_TRACK_BANDWIDTH
extern MPIBandwidthTracker tracker;
#define TRACK(type, bytes) tracker.add_count(type, bytes)
#define TRACK_NO_COUNT(type, bytes) tracker.add_count_no_track(type, bytes)
#else
#define TRACK(type, bytes)
#define TRACK_NO_COUNT(type, bytes)
#endif

/**
   MPIWrapper. This namespace provides a series of functions for doing "MPI
things" (tm) inside G6K. The primary goal here is to isolate all explicit MPI
functionality into this particular namespace, so that the sending / receiving
logic is somewhat isolated.

@remarks You should _not_ implement any functions in this file. All
implementations should be in various .cpp files.
@remarks This namespace requires some knowledge of G6K internals. This makes it
difficult to do certain things, as G6K is also required to know about this
namespace: this would cause hard-to-fix circular dependencies.
To fix this, we implement all functionality in mpi_wrapper.cpp. This still means
we have an include order dependency, but it works. The G6K layer should instead
deal with MPIObj.

@remarks This namespace accepts pointer parameters for outgoing messages as
const pointers. This differs from the MPI 1 and 2 specs, where source
buffers are expected to be non-const pointers. In MPI-3, in-place data
transformations (e.g for endianness) were outlawed, even in broadcasts.
However, this cannot possibly be checked without changing the syntax
dramatically of MPI, as (for example) in a broadcast/receiver case the calling
syntax is the same: \code{.cpp} MPI_Bcast(buffer, ...); // Called by both
parties! \endcode{}

However, some parts of G6K use const variables. To make it easier for
everyone, we have accept const pointers here and cast them to non-const buffers
in the function bodies. This is admittedly rather confusing, but this seems to
be the sweet spot. See https://www.mpi-forum.org/docs/mpi-3.1/mpi31-reportbw.pdf
(Page 148, Line 42) for rationale behind why this is OK for broadcast
operations.

@remarks Please note that every function in this namespace assumes that the
`comm` argument is an intracommunicator and not an intercommunicator. In other
words, we expect that `comm` connects many ranks in the same MPI group, and not
across multiple MPI groups. If you pass in a `comm` that is an intercommunicator
you will encounter undefined behaviour.

@remarks Please note that every function in this namespace that accepts a
MPI_Datatype will never free the result.

@remarks Please note that every function in this namespace (except from
set_root_rank_to_global_root_rank) expects you to have a communicator
where the root rank is MPIWrapper::global_root_rank.

@remarks Please note that this namespace uses the term "process" for MPI
proceses. For the unfamiliar, this is essentially the same as a "node" on the
network in our use case, although this is not necessarily generally true.

@remarks This namespace has a subtle limitation: for those not familiar with
MPI versions 1-3, MPI uses ints as a vocabulary type for almost everything. The
upshot of this is that without some relatively complicated handling it typically
isn't possible to send more than 4GB of data in one call: instead, multiple
calls need to be interleaved to send messages larger than 4GB in one go. This
requires some care. One effect of this is that certain functions in this
namespace may behave in unexpected ways: where this is the case, the function
documentation will spell this out.

In general, though, we do not expect this to be a problem that occurs regularly,
or even at all.

Intuitively, this is because the size of the buckets used inside G6K are rather
small compared to the database size. For example, in BGJ1 the buckets have size
"about the square root of the size of the database (asymptotically, α^2 → 1 −
√3/4 ≈ 0.3662)." This means that for us to need to send more than 4GB in one go,
we would need our sieving dimension `n` to be about 210 (under some
assumptions), which is far beyond the reach of modern architectures. Algorithms
with more aggressive bucketing strategies (e.g BDGL) would require an even
larger sieving dimension for this to become an issue.

If this is an issue, it wouldn't be impossible to use MPI-4 routines by simply
switching from the older MPI routines (e.g MPI_Recv) to the newer routines
(MPI_Recv_c). We haven't done this to allow for the possibility that some
users may want to use MPI-3 implementations. Also, MPI-4 is somewhat new by the
standards of most stable package managers (e.g Debian Stable does not yet have
it).

@remarks Understanding MPI can be challenging if you aren't used to the
programming model. Thankfully, there's a plethora of resources available both in
print form and online if you're interested. Here's a subset (please note that
almost all of these solely describe MPI-3 and not MPI-4):

- Using MPI, Portable Parallel Programming with the Messaging-Passing Interface,
  Gropp, Lusk, Skjellum
- Using Advanced MPI, Modern Features of the Message-Passing Interface,
  Groupp, Hoefler, Thakur, Lusk
- Rookie HPC (mpi): https://rookiehpc.github.io/mpi/
- A Comprehensive MPI Tutorial Resource: https://mpitutorial.com/
- MPI: A Message-Passing Interface Standard, Version 3.0,
https://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf (actually surprisingly
readable).
- MPI: A Message-Passing Interface Standard, Version 4.0,
https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf (also quite readable).
- Parallel Programming for Science and Engineering
  The Art of HPC, volume 2, Victor Eijkhout,
  https://web.corral.tacc.utexas.edu/CompEdu/pdf/pcse/EijkhoutParallelProgramming.pdf

**/
namespace MPIWrapper {

/**
 global_root_rank. This contains the global root rank of all MPI programs
 _after_ set_root_rank_to_global_root_rank has been called. This exists solely
 to allow us to have a one-stop shop where this can be configured.

 Note that this is not necessarily true if set_root_rank_to_zero has not
 been called: you will also need to update the communicator you're using
 to reflect this fact.

 Note that this value being 0 is the best choice for various reasons.
**/
static constexpr int global_root_rank{0};

/**
   set_root_rank_to_global_root_rank. This function accepts an MPI communicator,
 `comm` and sets the rank of the root rank to global_root_rank. If the rank of
 the root rank is already global_root_rank, then this function returns the
 `comm` argument. Otherwise, this function creates a new MPI communicator where
 the rank of the root rank is global_root_rank and returns that. This is
 primarily useful for MPI_IN_PLACE operations that involve the root. This
 function does not throw.

   @param[in] rank: the rank of the calling process.
   @param[in] is_root: true if this rank is the root rank, false otherwise.
   @param[in] comm: the MPI communicator to use.
   @return a new MPI comm.
 **/
MPI_Comm set_root_rank_to_global_root_rank(const int rank, const bool is_root,
                                           MPI_Comm comm) noexcept;

/**
  send_gso. This function broadcasts the full GSO object from the root rank to
all other ranks using `comm`. This corresponds to sending the GSO object from
the rank that is running the Cython layer associated with G6K to all other
ranks. This function follows the Lakos Rule (see test_utils.hpp for more).

  @param[in] full_n: the dimension of the mu object. The receiver
  must square-root this to get the full dimension of the lattice (we do it like
  this to mimic the usage in G6K).
  @param[in] mu: the GSO object. Must not be null.
  @param[in] comm: the MPI communicator to use.

  @remarks Note that this function doesn't actually modify any parameters: this
is just a relic of how MPI defines its function signatures. As a result, we
force `mu` to be a constant pointer here and then const_cast it to a type that
MPI will accept. The `mu` parameter is _not_ modified. See the namespace doc.

**/
void send_gso(unsigned int full_n, const double *const mu,
              MPI_Comm comm) MPI_DIST_MAY_THROW;

/**
     receive_gso. This function receives a GSO object from the root rank
(on the `comm`) and returns the result in `gso`. This function will first read
the number of entries needed and then read the GSO object. This function never
throws.

     @param[in] gso: the gso object to overwrite.
     @param[in] comm: the communicator to use.
**/
void receive_gso(std::vector<double> &gso, MPI_Comm comm) noexcept;

/**
   receive_gso_no_header. This function receives just the GSO portion of the
   GSO message. This function exists to allow callers to read the context change
   header first before calling this function. This is typically used by the dist
siever. This function never throws.
   @param[in] size: the number of gso elements to receive.
   @param[in] gso: the gso object to overwrite.
   @param[in] comm: the communicator to use.
   @return a vector of doubles containing the GSO.
**/
void receive_gso_no_header(const unsigned size, std::vector<double> &gso,
                           MPI_Comm comm) noexcept;

/**
   send_status. This function sends the siever's status from the root rank to
 all other ranks on `comm`. This function essentially just does a broadcast.
 This function does not throw.
   @param[in] status: the status to send.
   @param[in] comm: the communicator to use.
**/
void send_status(unsigned status, MPI_Comm comm) noexcept;

/**
   receive_params. This function receives the starting sieving parameters from
   the root rank.
   This function overwrites the passed in `param`, but does not overwrite either
   the thread count.

   This function does not throw.

   @param[out] params: the parameters to overwrite.
   @param[out] seed: the location to store the seed.
   @param[in] comm: the base communicator.
**/
void receive_params(SieverParams &params, uint64_t &seed,
                    MPI_Comm comm) noexcept;

/**
   receive_initial_setup. This function reads the starting sieving parameters
   from the root rank and the gso object, storing the results in `params` and
   `gso` respectively. This function essentially merges receive_params and
   receive_gso to make life easier for callers. This function never throws.
   @param[out] params: the location to store the received params.
   @param[out] gso: the location to store the received gso object.
   @param[out] seed: the location to store the rng seed.
   @param[in] comm: the MPI communicator to use.
**/
void receive_initial_setup(SieverParams &params, std::vector<double> &gso,
                           uint64_t &seed, MPI_Comm comm) noexcept;

/**
 broadcast_params. This function sends the starting sieving parameters from
 the root rank to all other ranks.
 This function does not modify the passed in `param`, but MPI requires a
non-const pointer there for generality. This function does not throw.

For more details on the layout, see layouts.hpp.

 @param[in] params: the parameters to broadcast.
 @param[in] seed: the rng seed to broadcast.
 @param[in] comm: the base communicator.
**/
void broadcast_params(const SieverParams &params, const unsigned int long seed,
                      MPI_Comm comm) noexcept;

/**
   broadcast_el. This function broadcasts the extend left operation to all
   attached ranks from the root rank using `comm`. This function does not
throw.
   @param[in] lp: the extend left parameter.
   @param[in] comm: the communicator to use.
**/
void broadcast_el(const unsigned int lp, MPI_Comm comm) noexcept;

/**
 broadcast_er. This function broadcasts the extend right operation to all
 attached ranks from the root rank using `comm`. This function does not
throw.
 @param[in] rp: the extend right parameter.
 @param[in] comm: the communicator to use.
**/
void broadcast_er(const unsigned int rp, MPI_Comm comm) noexcept;
/**
 broadcast_sl. This function broadcasts the shrink left operation to all
 attached ranks from the root rank using `comm`. This function does not
throw.
 @param[in] lp: the shrink left parameter.
 @param[in] comm: the communicator to use.
**/
void broadcast_sl(const unsigned int lp, const bool down_sieve,
                  MPI_Comm comm) noexcept;

/**
   reduce_best_lifts_to_root. This function accepts a set of best lifts
(`lifts_in`) from the current rank and globally computes the shortest set of
best lifts. The result is stored in lifts_out if my_rank ==
MPIWrapper::global_root_rank. Otherwise, lifts_out is not modified. This
function can be called by both root ranks and non-root ranks.

   Internally this function uses a custom MPI operation to do a global reduction
across all of the inputs. This requires (amongst other things) re-arranging the
data so that everything works properly together.
   @param[in] lifts_in: the set of best lifts from this rank.
   @param[out] lifts_out: the set of output lifts from this rank.
   @param[in] full_n: the dimension of the fully lifted vectors.
   @param[in] my_rank: the rank of this rank.
   @param[in] op: the MPI operation to use for the reduction.
   @param[in] comm: the communicator to use.
**/
void reduce_best_lifts_to_root(const std::vector<LiftEntry> &lifts_in,
                               std::vector<LiftEntry> &lifts_out,
                               const unsigned full_n, const int my_rank,
                               MPI_Op op, MPI_Comm comm) noexcept;

/**
   send_topology. This function sends the topology from the root to all
   attached ranks. This function does not throw.
   @param[in] topology: the type of topology to use.
   @param[in] comm: the MPI communicator to use.
 **/
void send_topology(const DistSieverType topology, MPI_Comm comm) noexcept;

/**
   get_topology. This function receives the topology from the root, overwriting
   `topology`. This function does not throw.
   @param[out] topology: the type of topology to use.
   @param[in] comm: the MPI communicator to use.
 **/
void get_topology(DistSieverType &topology, MPI_Comm comm) noexcept;

/**
   gather_buckets. This function collects all of the thread arguments for each
   rank and stores them into `buckets_per_rank`. This function does not throw.
Note that this function can be called by any type of rank.
   @param[in] buckets: the number of buckets associated with this rank.
Overwritten by the maximum number of buckets requested globally.
@param[in] scale_factor: the amount to scale the number of buckets by.
@param[in] bucket_batches: the number of buckets to have running in parallel.
@param[in] comm: the MPI_Comm to use.
**/
void gather_buckets(uint64_t &buckets, unsigned &scale_factor,
                    unsigned &bucket_batches, unsigned &scratch_buffers,
                    MPI_Comm comm) noexcept;

/**
   collect_memory. This function collects the number of vectors that each rank
can hold at maximum and writes the results to `memory_per_rank`. In particular,
   the amount of memory held by rank `i` is written to `memory_per_rank[i]`.
   This function does not throw. This function can be called by any rank.

   @param[out] memory_per_rank: the location to store the received `memory per
rank`.
   @param[in] memory: the maximum number of vectors that the calling rank.
   @param[in] comm: the communicator to use.

**/
void collect_memory(std::vector<uint64_t> &memory_per_rank,
                    const uint64_t memory, MPI_Comm comm) noexcept;

/**
     get_full_database_size. This function collects the number of vectors in the
database of each rank and stores the results in `sizes`. This function does
not throw. Note that this function should only be called by root ranks: it is
undefined behaviour to call this from a non-root rank.

@param[out] sizes: the location to store the number of vectors held by each
rank.
@param[in] size: the number of vectors held by this rank.
@param[in] comm: the communicator to use.
**/
void get_full_database_size(std::vector<int> &sizes, const int size,
                            MPI_Comm comm) noexcept;

/**
   send_database_size_to_root. This function sends the size of this rank's
   database to the root rank. This function does not throw. Note that this
function should only be called by non-root ranks: it is undefined behaviour to
call this from a root rank. This function does not throw.
   @param[in] size: the size of this rank's database as an int.
   @param[in] comm: the MPI_Comm to use.
**/
void send_database_size_to_root(const int total, MPI_Comm comm) noexcept;

/**
   send_database_to_root. This function sends every Entry in `db` to the root
 rank. This function does not throw. Note that it is undefined behaviour to call
 this from  a root rank.
   @param[in] db: the vectors to send.
   @param[in] n: the number of elements per vector.
   @param[in] l: the (inclusive) lower bound for where to start sending
 entries.
   @param[in] r: the (exclusive) upper bound for where to stop sending entries.
   @param[in] comm: the MPI comm to use.
 **/
void send_database_to_root(const std::vector<Entry> &db, const unsigned n,
                           MPI_Comm comm) noexcept;

/**
   get_database. This function collects every vector from every non-root rank
and stores the results in `db`. This function may resize `db` if `db` is not
large enough to hold the results, which may result in allocation. This function
does not throw.
   @param[out] db: the location to store the vectors.
   @param[in] n: the number of elements per vector.
   @param[in] comm: the MPI comm to use.
   @remarks The vectors read from non-root ranks are appended after the current
entries in db.
**/
void get_database(std::vector<Entry> &db, const unsigned n,
                  MPI_Comm comm) noexcept;

/**
   split_database. This function evenly divides the sieving database `db`
between all ranks in the cluster (each having space for `memory_per_rank[i]`
vectors at most).

   Briefly, the algorithm used by this function is a weighted splitting
algorithm. We assign a share of the `db` to each rank depending on the total
fraction of the memory they have:

   \code{.cpp}
       // The portion for rank `i`.
       const auto share = (double)memory_per_rank[i] /
std::accumulate(memory_per_rank.begin(), memory_per_rank.end(), 0); \endcode{}

   Since this approach may cause rounding errors, we explicitly round down all
shares to ensure that the database is under-provisioned. We then send the extra
vectors to the rank who has the most storage: this process is deterministic.

   Note that this function may not do what you want, as it simply splits the
database evenly. This is essentially a random data partitioning, because the
database is not sorted. If you do want some sort of guarantee, see
split_database_ordered instead.

   This function must not be called by a non-root rank: otherwise, the behaviour
is undefined.

   This function does not throw.
   @param[in] db: the sieving database to split.
   @param[in] n: the number of elements per vector.
   @param[in] memory_per_rank: the number of vectors that each rank can hold at
most.
   @param[in] comm: the MPI communicator to use.
   @remarks Note that after this function is called, `db` will only contain the
vectors that the root rank has been allocated. This will mean that the `db` size
is smaller.


@remarks Note that this function is has a subtle quirk: namely, whilst the
memory_per_rank is a 64-bit unsigned value, we in fact only use around 32-bits
of each of these. The reason for this is complicated: see the namespace doc for
more. In any case, to guard against this we assert that the number of vectors
allocated to each rank is at most std::numeric_limits<int>::max(), which guards
us against this failure.
**/
void split_database(std::vector<Entry> &db, const unsigned n,
                    const std::vector<uint64_t> &memory_per_rank,
                    MPI_Comm comm) noexcept;
/**
   receive_split_database. This function receives the share of the sieving
database from the `split_database` call for this rank. This function does not
throw. Note that this function must not be called by a root rank: otherwise, the
behaviour is undefined.

Nothing in this database is initialised after the call. Instead, the caller will
need to initialise the database entries directly.

   @param[out] db: the memory to store the sieving database. This will be
resized and (likely) overwritten. Do not rely on this being the same between
calls!
@param[in] n: the number of elements per vector.
 @param[in] comm: the MPI communicator to use.
**/
void receive_split_database(std::vector<Entry> &db, const unsigned n,
                            MPI_Comm comm) noexcept;

/**
   split_database_ordered. This function accepts a (cdb, db) pair and splits the
database evenly amongst all attached ranks, depending on their memory
requirements. Each rank is guaranteed to receive a database segment they can
hold.

This function operates similarly to split_database in almost all ways. The key
difference is that the database is divided according to cdb, rather than
according to db. This is done primarily to allow for us to have some sort of
guarantees on the data partioning, which makes certain distributed computing
problems easier. All other guarantees given in split_database also apply here.

This function does not throw. Note that this function must be called by a root
rank, otherwise the result is undefined.

@param[in] cdb: the Compressed database to send. Should be sorted for reasonable
guarantees, but this is not checked in this function.
@param[in] db: the database to send.
@param[in] n: the  number of elements per vector.
@param[in] memory_per_rank: the number of vectors that each rank can hold at
most.
@param[in] comm: the MPI communicator to use.
@remarks After this function is called, both `cdb` and `db` will be smaller,
containing only the vectors that the root rank has been allocated.
**/
void split_database_ordered(std::vector<CompressedEntry> &cdb,
                            std::vector<Entry> &db, const unsigned n,
                            const std::vector<uint64_t> &memory_per_rank,
                            MPI_Comm comm) noexcept;

/**
   receive_database_ordered. This function accepts a (cdb, db) pair and stores
   a received database share from the root in them. In particular, this function
   receives a partitioned share of the database. For simplicity, this function
   inserts everything linearly: in other words, cdb[i].i == i.

   This function is the pair of split_database_ordered. All other guarantees are
inherited from that function. Note that, as with `receive_database`, the entries
in the returned database are not fully computed, and so the caller must call the
relevant functions in G6K to properly set these values up.


   Note that it is undefined behaviour to call this function from a root rank.
   This function does not throw.
   @param[out] cdb: the Compressed database to overwrite.
   @param[out] db: the database to overwrite.
   @param[in[ n: the number of elements per vector.
   @param[in] comm: the MPI comm to use.
**/
void receive_database_ordered(std::vector<CompressedEntry> &cdb,
                              std::vector<Entry> &db, const unsigned n,
                              MPI_Comm comm) noexcept;

/**
     split_database_uids. This function splits the `db` argument across the
entire cluster depending on the rules specified in `map`. In more detail, this
function iterates over the `db` and sends the vectors to the node that owns that
portion of the hash table. This function removes any sent vectors from the db
and cdb, resizing both to only contain vectors that belong to this node. Note
that this function can only be called by the root process. This function does
not throw.
     @param[in] cdb: the cdb of this process.
     @param[in] db: the db of this process.
     @param[in] map: the slot map of this process.
     @param[in] n: the dimension of the outgoing vectors.
     @param[in] comm: the MPI communicator to use.
**/
void split_database_uids(std::vector<CompressedEntry> &cdb,
                         std::vector<Entry> &db,
                         const std::array<int, DB_UID_SPLIT> &map,
                         const unsigned n, MPI_Comm comm) noexcept;
/**
     receive_database_uids. This function receives the database segment produced
by the root calling split_database_uids and inserts it into `cdb` and `db`. This
function does not throw.
     @param[in] cdb: the cdb of this process. Must be empty.
     @param[in] db: the db of this process. Must be empty.
     @param[in] n: the dimension of the incoming vectors.
     @param[in] comm: the MPI communicator to use.
**/
void receive_database_uids(std::vector<CompressedEntry> &cdb,
                           std::vector<Entry> &db, const unsigned n,
                           MPI_Comm comm) noexcept;

/**
   db_size. This function accepts a `size` parameter and gathers the size of the
   database across all processes, returning the result. This function does not
throw.
   @param[in] size: the size of the database held by this process.
   @param[in] comm: the MPI comm to use.
   @return the global size of the database.
**/
size_t db_size(const size_t size, MPI_Comm comm) noexcept;

/**
 global_saturation. This function accepts a `sat` parameter and gathers the
saturation count of the database globally, returning the result. This function
does not throw.
 @param[in] sat: the number of vectors under the saturation radius for this
rank.
 @param[in] comm: the MPI comm to use.
 @return the global number of vectors under the saturation radius.
**/
size_t global_saturation(const size_t sat, MPI_Comm comm) noexcept;

/**
   grow_db. This function issues a grow call to all attached ranks. This has the
effect of growing the global database to contain `N` elements. This function
fairly splits the elements amongst each attached rank depending on
`memory_per_rank`. This function does not throw. Please note that this function
does not grow the database for _this_ rank: as a result, the caller must resize
their database after the call.
   @param[in] N: the size of the database after growing.
   @param[in] large: the large parameter from G6K.
   @param[in] memory_per_rank: the available memory per rank.
   @param[in] comm: the MPI comm to use.
   @return the size of the database for this rank.
**/
size_t grow_db(const size_t N, const unsigned large,
               const std::vector<uint64_t> &memory_per_rank,
               MPI_Comm comm) noexcept;

/**
   reserve_db. This function issues a reserve call to all attached ranks. This
has the effect of growing the global database to be able to contain at least `N`
elements. This function fairly splits the elements amongst each attached rank
depending on `memory_per_rank`. This follows essentially the same procedure as
grow_db. This function does not throw. Please note that this function does not
grow the database for _this_ rank: as a result, the caller must resize their
database after the call.
   @param[in] N: the size of the database after growing.
   @param[in] memory_per_rank: the available memory per rank.
   @param[in] comm: the MPI comm to use.
   @return the size of the database for this rank.
**/
size_t reserve_db(const size_t N, const std::vector<uint64_t> &memory_per_rank,
                  MPI_Comm comm) noexcept;

/**
     shrink_db. This function issues a shrink call to all attached ranks. This
has the effect of shrinking the global database to contain `N` elements. This
function fairly splits the shrinking amongst the entire database by working out
the norm of the `N`-th element across the entire database using a binary search.
This function does not throw. Please note that this function does not shrink the
database for _this_ rank: as a result, the caller must resize their database
after the call.
     @param[in] N: The size of the database after shrinking.
     @param[in] min: the minimum norm of entries in this database.
     @param[in] max: the maximum norm of entries in this database.
     @param[in] cdb: the CDB for this rank. Must be sorted.
     @param[in] comm: the communicator to use.
     @return the size of the database for this rank after shrinking.
**/
size_t shrink_db(const size_t N, const FT min, const FT max,
                 const std::vector<CompressedEntry> &cdb,
                 MPI_Comm comm) noexcept;

/**
   sort_db. This function issues a sort call to all attached ranks. This has the
effect of making each rank locally sort their cdbs. This function does not
throw.
   @param[in] comm: the MPI Comm to use.
**/
void sort_db(MPI_Comm comm) noexcept;

/**
   receive_command. This function receives a command from the root rank,
returning the result in an array. The first returned value corresponds to the
type of message, whereas the second contains the parameter for the operation.
This function does not throw.
   @param[in] comm: the MPI_Comm to use.
   @returns a new change status command at position 0, along with potentially
useful information in position 1.
**/
std::array<unsigned, 2> receive_command(MPI_Comm comm) noexcept;

/**
   gso_update_postprocessing. This function sends a post-processing message to
all attached ranks from the root rank. This message informs each attached rank
that they need to run a gso update. This function does not throw.
   @param[in] l_: the new l_ value.
   @param[in] r_: the new r_ value.
   @param[in] old_n: the old n value.
   @param[in] M: the change-of-basis matrix. This matrix is passed in as a 1D
C-style array. Note that this matrix must be of size (r_ * l_) * old_n.
@param[in] should_redist: true if we need to call the redistribution function,
false otherwise.
@param[in] comm: the MPI communicator to use.
**/
void gso_update_postprocessing(const unsigned int l_, const unsigned int r_,
                               const unsigned old_n, long const *M,
                               const bool should_redist,
                               MPI_Comm comm) noexcept;

/**
   receive_gso_update_postprocessing. This function receives a post-processing
message _payload_ from the root rank. In particular, this function receives the
new dimensions (l_, r_), the transformation matrix M, and a flag that tells us
if we should redistribute or not (`should_redist`). This function should only be
called after an appropriate call to receive_command has been issued that has
indicated calling this is necessary. This function returns (l_, r_,
should_redist) in an array.

   @param[out] M: the vector to contain the transformation matrix after this
function is called.
   @param[in] old_n: the previous value of `n`. This is passed in to help resize
M.
   @param[in] comm: the MPI communicator to use.
   @return (l_, r_, should_redist) in an array.
   @remarks The (l_, r_, should_redist) is returned in an array to prevent the
compiler from believing that the two out parameters might alias. We could use
__restrict__ for this, but it isn't necessarily that widely supported.
**/
std::array<unsigned, 3>
receive_gso_update_postprocessing(std::vector<long> &M, const unsigned old_n,
                                  MPI_Comm comm) noexcept;

/**
   send_stop. This function sends a stop message from the root rank to all other
ranks. This function does not throw. This function must not be called by
non-stopping ranks.
   @param[in] comm: the MPI communicator to use.
**/
void send_stop(MPI_Comm comm) noexcept;

/**
   start_bgj1. This function forwards a message to all attached ranks to tell
them it is time to start bgj1 sieving. This function does not throw.
   @param[in] alpha: the alpha parameter for sieving.
   @param[in] comm: the MPI communicator to use.
**/
void start_bgj1(const double alpha, MPI_Comm comm) noexcept;
/**
   receive_alpha. This function reads the value of `alpha` from the root rank.
This corresponds to receiving the alpha value from start_bgj1.
   @param[in] comm: the MPI communicator to use.
   @return the value of alpha.
**/
double receive_alpha(MPI_Comm comm) noexcept;

/**
   is_one_g6k. This function accepts an `is_g6k` argument and returns true if
any rank on the network is registered as g6k. This is mainly meant to circumvent
issues around the dist siever being launched without a corresponding G6K process
somewhere. This function never throws.
   @param[in] is_g6k: true if this rank is g6k, false otherwise.
   @param[in] comm: the MPI communicator to use.
   @return true if one rank on the network is G6K, false otherwise.
**/
bool is_one_g6k(const bool is_g6k, MPI_Comm comm) noexcept;

/**
   broadcast_initialize_local. This function broadcasts an IL message to all
 attached ranks. This function never throws.
   @param[in] ll: the ll_ argument to initialize_local.
   @param[in] l: the l_ argument to initialize_local.
   @param[in] r: the r_ argument to initialize_local.
   @param[in] comm: the MPI communicator to use.
 **/
void broadcast_initialize_local(const unsigned int ll, const unsigned int l,
                                const unsigned int r, MPI_Comm comm) noexcept;

/**
   receive_l_and_r. This function receives the l and r parameters for
 initialize_local from the root rank and returns the result. This function never
 throws.
   @param[in] comm: the MPI comm to use.
   @return an array containing l at the first index and r at the second index.
 **/
std::array<unsigned, 2> receive_l_and_r(MPI_Comm comm) noexcept;

/**
   build_global_histo. This function builds the histogram of norms for G6K to
consume. If the caller of this function is a root rank, then this function
stores the resulting histogram in the `histo` argument.  This function operates
across all ranks in the cluster. This function does not throw.
   @param histo the location to store the histogram. Must be pre-populated by
the caller. Note that this function assumes that histo is a pointer to a G6K
histogram.
   @param[in] comm: the MPI communicator to use.
**/
void build_global_histo(long *const histo, MPI_Comm comm) noexcept;

/**
   broadcast_build_histo. This function forwards a message from the root to all
attached ranks to tell them to build their histograms. This function never
throws. Note that it is undefined behaviour to call this function from a
non-root rank.
   @param[in] comm: the MPI communicator to use.
**/
void broadcast_build_histo(MPI_Comm comm) noexcept;

/**
   broadcast_db_size. This function broadcasts a message from the root to all
   attached ranks to tell them to provide the size of their database. This
function never throws. Note that it is undefined behaviour to call this function
from a non-root rank.
   @param[in] comm: the MPI communicator to use.
**/
void broadcast_db_size(MPI_Comm comm) noexcept;
void broadcast_db_capacity(MPI_Comm comm) noexcept;

/**
   gather_gbl_sat_variance. This function computes the population variance of
GBL_sat_count across all attached ranks and returns the result to the root rank.
For all non-root ranks, the function returns 0.0. This function never throws.
   @param[in] cur_sat: the saturation count for this rank.
   @param[in] comm: the MPI communicator to use.
   @return the variance for the saturation count across all ranks or 0.0 if the
caller is a non-root rank.
**/
double gather_gbl_sat_variance(const size_t cur_sat, MPI_Comm comm) noexcept;

/**
 gather_gbl_ml_variance. This function computes the population variance of
GBL_max_len across all attached ranks and returns the result to the root rank.
For all non-root ranks, the function returns 0.0. This function never throws.
 @param[in] cur_gbl_ml: the GBL_max_len for this rank.
 @param[in] comm: the MPI communicator to use.
 @return the variance for the GBL_max_len across all ranks or 0.0 if the caller
is a non-root rank.
**/
double gather_gbl_ml_variance(const double cur_gbl_ml, MPI_Comm comm) noexcept;

/**
   get_best_lifts_op. This function creates a new MPI_Op for sending the best
 lifts to the root rank and returns the result. This function does not throw.
 Note that a new MPI operation is created each time this function is called:
 there's no caching with this function.
   @return a new MPI_Op that can be used for gathering the best lifts.
 **/
MPI_Op get_best_lifts_op() noexcept;

/**
   serialise_stats. This function serialises the `stats` argument and sends the
result to the root rank. This function does not throw. Note that it is undefined
behaviour to call this function with a root rank.
   @param[in] stats: the stats object to serialise.
   @param[in] comm: the MPI communicator to use.
   @remarks This function only serialises the unsigned long stats at present.
**/
void serialise_stats(const SieveStatistics &stats, MPI_Comm comm) noexcept;

/**
 deserialise_stats_and_add. This function receives the SieveStatistics object
from each rank on the network and adds the received stats counts to the `stats`
argument. This function does not throw. Note that this function can only be
called by a root rank.
 @param[out] stats: the stats object that stores all stats on the network.
 @param[in] comm: the MPI communicator to use.
 @remarks This function only collects the unsigned long stats at present.
**/
void deserialise_stats_and_add(SieveStatistics &stats, MPI_Comm comm) noexcept;

/**
     reset_stats. This function broadcasts a reset stats message from the root
rank to all attached processes. This message has the affect of causing all nodes
to clear their stats counters. This function does not throw. Note that it is
undefined behaviour to call this function from a non-root process.
     @param[in] comm: the MPI communicator to use.
**/
void reset_stats(MPI_Comm comm) noexcept;

/**
   get_min_max_len. This function returns the minimum of all `max_len` arguments
 passed in across the network. This function does not throw.
   @param[in] max_len: the maximum length for this node.
   @param[in] comm: the MPI communicator to use.
   @return the minimum `max_len` argument supplied globally.
 **/
double get_min_max_len(double max_len, MPI_Comm comm) noexcept;

/**
   setup_slot_map. This function divides up the uid_hash_table's slots
depending on the values in memory_per_rank. This function applies an even split:
we divide up the hash table depending on how much free memory each rank has.
This function does not throw.
   @param[out] map: the slot to node mapping.
   @param[in] memory_per_rank: the amount of memory held by each rank.
**/
void setup_slot_map(std::array<int, DB_UID_SPLIT> &map,
                    const std::vector<uint64_t> &memory_per_rank) noexcept;

std::vector<int>
setup_owner_array(std::vector<uint32_t> &owner,
                  const std::vector<uint64_t> &memory_per_rank) noexcept;

/**
     redistribute_database. This function globally re-arranges the database.
  Briefly, this function iterates over the `db` and forwards any entries that
  don't belong to this node to the correct home, removing them from the cdb, the
  db, and the hash_table. In exchange, this node receives any vectors from
  elsewhere on the network that belong to this node. This function does not
  throw. This function returns any duplicates in the database that are received
  during this process. Each node must call this function at the end of collision
  checks.
     @param[in] db: the db for this node.
     @param[in] cdb: the cdb for this node.
     @param[in] map: the map of slots to nodes. This must have been initialised
  by the caller and must be the same across all nodes for correctness.
     @param[in] n: the dimension of received vectors.
     @param[in] hash_table: the hash table. All sent vectors are removed from
  the hash table, and all incoming ones are added.
     @param[in] comm: the MPI communicator to use.
     @param[out] duplicates: a vector containing any duplicates. These are cdb
  indices.
     @param[out] incoming: a vector containing entries that were read in. These
  are cdb indices.
  **/
void redistribute_database(std::vector<Entry> &db,
                           std::vector<CompressedEntry> &cdb,
                           const std::array<int, DB_UID_SPLIT> &slot_map,
                           const unsigned n, UidHashTable &hash_table,
                           MPI_Comm comm, std::vector<unsigned> &duplicates,
                           std::vector<unsigned> &incoming) noexcept;

void grab_lift_bounds(std::vector<FT> &lift_bounds, FT &lift_max_bound,
                      MPI_Comm comm) noexcept;

FT get_norm(const size_t N, const FT min, const FT max,
            const std::vector<CompressedEntry> &cdb, const ContextChange tag,
            MPI_Comm comm) noexcept;

void forward_and_gather_buckets(
    const unsigned n, const unsigned buckets_per_rank,
    const std::vector<std::vector<std::vector<unsigned>>> &cbuckets,
    const std::vector<Entry> &db, const std::vector<int> &sizes,
    std::vector<ZT> &incoming_buffer, std::vector<ZT> &outgoing_buffer,
    std::vector<bucket_pair> &bucket_pairs, std::vector<int> &scounts,
    std::vector<int> &sdispls, std::vector<int> &rcounts,
    std::vector<int> &rdispls, const MPI_Datatype entry_type, MPI_Comm comm,
    MPI_Request *request) noexcept;

long double get_cpu_time(MPI_Comm comm, std::clock_t time) noexcept;
uint64_t get_extra_memory_used(MPI_Comm comm, uint64_t extra_used) noexcept;

uint64_t get_total_bandwidth(MPI_Comm comm) noexcept;
uint64_t get_total_messages(MPI_Comm comm) noexcept;
uint64_t get_messages_for(const ContextChange type, MPI_Comm comm) noexcept;
uint64_t get_bandwidth_for(const ContextChange type, MPI_Comm comm) noexcept;

std::array<uint64_t, 2> get_unique_ratio(const uint64_t nr_uniques,
                                         const uint64_t nr_sends,
                                         MPI_Comm comm) noexcept;

uint64_t get_adjust_timings(const uint64_t time, MPI_Comm comm) noexcept;

void reset_bandwidth(MPI_Comm comm) noexcept;

std::vector<unsigned>
bdgl_exchange(std::vector<Entry> &db, std::vector<CompressedEntry> &cdb,
              const unsigned n, std::vector<size_t> &sizes,
              std::vector<uint32_t> &buckets, const size_t bsize,
              const std::vector<uint32_t> &owner,
              const std::vector<uint32_t> &ours, MPI_Comm comm);

void start_bdgl(const size_t nr_buckets_aim, const size_t blocks,
                const size_t multi_hash, MPI_Comm comm) noexcept;

std::array<uint64_t, 3> receive_bdgl_params(MPI_Comm comm) noexcept;
} // namespace MPIWrapper

#endif
