#ifndef INCLUDED_MPI_OBJ_HPP
#define INCLUDED_MPI_OBJ_HPP

#include "bucket_pair.hpp"
#include "constants.hpp"
#include "siever_types.hpp"
#include "thread_pool.hpp"

#ifdef G6K_MPI
#include "context_change.hpp"
#include "topology.hpp" // Needed for topology things.
#include "uid_hash_table.hpp"
#include <mpi.h> // Only if G6K_MPI is used.
#endif

#include "bgj1_bucketing_interface.hpp"
#include "communicator.hpp"
#include "compat.hpp"
#include "entry.hpp"
#include "mpi_bandwidth_tracker.hpp"
#include "mpi_request.hpp"
#include "mpi_timer.hpp"
#include "packed_sync_struct.hpp"
#include "test_utils.hpp" // Needed for conditional throwing.

#include <array> // Needed for std::array.
#include <atomic>
#include <cassert>
#include <cstdint> // Needed for int64_t.
#include <iostream>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector> // Needed for std::vector.

// Needed for forward declarations in function signatures.
class SieverParams;
class Siever;
class SieveStatistics;
class UidHashTable;
struct LiftEntry;
struct CompressedEntry;
struct QEntry;
class ProductLSH;

// We conditionally let certain functions be constexpr if MPI isn't used
// This is to make it easier for the compiler to optimise away calls that we
// don't actually use.
#ifdef G6K_MPI
#define G6K_MPI_CONDITIONAL_CONSTEXPR
#else
#define G6K_MPI_CONDITIONAL_CONSTEXPR
#endif

/**
   MPIObj.
   This class solely exists to make it easier for G6K to interact with
   MPI, without necessarily needing to include lots of parts of the MPI wrapper
   directly. In particular, this class exists solely to abstract away needing
to store any MPI state: namely, all communicator information etc is stored
inside this class.

   Please note that this class is only conditionally thread safe: certain
guarantees are made on some areas of the class. These will be highlighted in the
documentation of the e.g functions and variables.

   Note that this struct exhibits quite strong conditional behaviour.
   If G6K's build system (or another compiler pragma) defines G6K_MPI, then MPI
is used. Otherwise, the functions in this struct don't do anything at all (and
nor does the struct store anything). This is primarily to make it easier to use
G6K without the extra MPI stuff included (it might not always be advantageous to
do so).

  This class makes rather strong use of MPI persistent requests and non-blocking
I/O. This means that the class has quite a complicated API: when in doubt, read
the documentation.

  @remarks Extending the functionality here is best done by providing a type of
Topology. This is held inside this object and provides a unified interface for
the caller (i.e all calls come through the MPI object).

  @remarks As with elsewhere in this project, this class implicitly associates
threading with memory capacity. This is primarily to prevent ourselves from
needing to use more complicated memory strategies.
**/
class MPIObj {
public:
  /**
     State. This enum contains the various states that the MPIObj may be in.
     This is typically used for e.g preventing certain operations from being
  carried out in certain contexts.


  The state machine for this enum is as follows. We start in the DEFAULT state.
  A caller may switch to any other state. When switching out of any other state,
  the state must be reset to DEFAULT. This mimics how things are done in G6K
  more broadly.
  **/
  enum class State : uint8_t {

    // DEFAULT. If the MPI object is in the DEFAULT state, then it is not
    // carrying out
    // any of the operations indicated by other members of this enum. In this
    // case,
    // the only useful question is if the sieve is active or not, which is
    // stored elsewhere.
    DEFAULT,

    // SIEVING. If the MPI object is in the SIEVING state, then it means that
    // the root process
    // has sent a BGJ1/BDGL message to attached ranks. This state prevents the
    // root process from sending
    // any context change messages (e.g EL/ER/SL) or any SORT messages.
    SIEVING,

    // CONTEXT_CHANGE. If the MPI object is in the CONTEXT_CHANGE state then
    // this means that the root process
    // is in a call to SL/EL/ER. This state prevents the root process from
    // sending multiple additional e.g
    // initialize_local messages, which may be expensive.
    CONTEXT_CHANGE,

    // RESIZING. If the MPI object is in the RESIZING state then this means that
    // the root process
    // is in a call to (grow|shrink)_db. This state prevents the root process
    // from sending multiple
    // additional (grow|shrink) calls.
    RESIZING,
  };

  /**
     SyncChange. This struct is used to represent changes to the saturation and
  trial count globally. This is issued at the beginning of each bgj1 bucketing
  batch.
  **/
  struct SyncChange {
    uint64_t sat_drop;
    uint64_t any_sub_zero;
  };

private:
#ifdef G6K_MPI
  /**
     stopped. This variable is true if the stop barrier has been completed and
  false otherwise.
  **/
  bool stopped;

public:
  /**
     issued_stop. This variable is true if the stop barrier has been issued to
  signal the end of the sieve and false otherwise.
  **/
  bool issued_stop;

private:
  /**
     is_barrier_empty. This varialbe is true if the stop barrier is not in use
  and false otherwise.
  **/
  bool is_barrier_empty;

  /**
     is_used. This is true if the wait_head is in use and false otherwise.
  **/
  bool is_used;

  /**
     wait_head. This is the index for the next size request that can be upgraded
  to a bucketing reply. This is loosely synchronised across all ranks.
  **/
  unsigned wait_head;

  /**
     comm. This contains the current communicator of this object.
  **/
  MPI_Comm comm;

  /**
     rank. This contains the rank of this node.
   **/
  int rank;

  /**
     topology. This contains a unique pointer to a topology object.
  **/
  std::unique_ptr<Topology> topology;

  /**
     buckets_per_rank. This contains the number of buckets that are issued by
     each rank in each iteration. This is fixed across all ranks to allow for
  slightly more efficient MPI communications.
  **/
  uint64_t buckets_per_rank;

  /**
     memory_per_rank. This contains the maximum number of vectors that each rank
     may hold. In particular, rank i has its maximum number of vectors stored at
  index i. This is primarily for evenly splitting up the database. Note that
  this is not used in non-root ranks.
  **/
  std::vector<uint64_t> memory_per_rank;

  /**
     total_bucket_count. This contains the total bucket count for the network.
   This is cached for better performance. All ranks use this field to decrement
   their trial counter in sieves that use this (e.g bgj1).
   **/
  uint64_t total_bucket_count;

  /**
     requests. This object contains various requests that are frequently checked
  on each iteration. This object exists to make these requests quicker to check.
  **/
  MPIRequest requests;

  std::vector<std::vector<bucket_pair>> bucket_pairs;

  struct alignas(2 * CACHELINE_ALIGNMENT) inner_outgoing_vector {
    std::mutex lock;
    std::vector<ZT> buffer;

    size_t bytes_used() const noexcept {
      return buffer.capacity() * sizeof(ZT);
    }
  };

  struct alignas(2 * CACHELINE_ALIGNMENT) outgoing_vector {
    std::vector<inner_outgoing_vector> buffers;

    size_t bytes_used() const noexcept {
      const auto top = buffers.capacity() * sizeof(inner_outgoing_vector);
      size_t size{};
      for (auto &v : buffers) {
        size += v.bytes_used();
      }
      return size + top;
    }
  };

  std::vector<outgoing_vector> tmp_insertion_buf;

  struct alignas(2 * CACHELINE_ALIGNMENT) insertion_vector {
    std::vector<ZT> outgoing_buffer;
    std::vector<ZT> incoming_buffer;

    size_t bytes_used() const noexcept {
      return sizeof(ZT) *
             (outgoing_buffer.capacity() + incoming_buffer.capacity());
    }
  };

  struct insertion_entry {
    uint64_t global_queue_pos;
    unsigned thread_id;
    unsigned thread_queue_pos;
  };

  std::vector<ZT> bdgl_send_buf, bdgl_recv_buf;

  /**
     insertion_vector_bufs. This vector contains scratch storage for
  writing and receiving insertion batches. Note that these are only ever
  accessed from a single thread.

  In particular, we allocate `bucket_batches` many inner vectors and pack
  outgoing elements. The number of outgoing elements is made available via
  a previous send.
  **/
  std::vector<insertion_vector> insertion_vector_bufs;

  /**
   state. This variable contains the state of this MPI object. See the
   declaration of the enum for more. Note that this is mutable to allow for
usage in e.g database change functions (this is to handle recursive dispatch).
**/
  mutable State state;

  /**
     active. This variable is true if the G6K database is distributed across
   multiple ranks and false otherwise.
  **/
  bool active;

  /**
     reduce_best_lifts_op. This variable contains the MPI operation for
   collecting the best lifts at the root rank. This is cached to prevent
   needing to recreate this op each time we collect the best lifts.
   **/
  MPI_Op reduce_best_lifts_op;

public:
  // Each thread has a unique set of (cbucket, transaction_db, bucket) that is
  // keyed by its thread ID. This exists in this object (and not, say, in
  // Siever) simply so that it isn't included if G6K is compiled without MPI
  // support.
  struct alignas(CACHELINE_ALIGNMENT * 2) ThreadEntry {
    std::vector<CompressedVector> cbucket;
    std::vector<Entry> transaction_db;
    std::vector<Entry> bucket;
  };

  // In BDGL, each thread has a unique set of (cbucket, bucket, queue)
  // that is used for sieving. This (similarly to the ThreadEntry) is keyed
  // based on the thread's ID.
  struct alignas(CACHELINE_ALIGNMENT * 2) BdglThreadEntry {
    std::vector<CompressedVector> cbucket;
    std::vector<Entry> bucket;
    std::vector<Reduction> t_queue;
    std::vector<ZT> insert_db;
  };

private:
  struct BdglAux {
    std::vector<std::vector<std::vector<unsigned>>> cbuckets;
    std::vector<int> sizes;
  };

  BdglAux bdgl_aux;

  std::vector<ThreadEntry> thread_entries;
  std::vector<BdglThreadEntry> bdgl_thread_entries;

  // Note: these are padded as well because perf c2c showed that there was some
  // false sharing on these. The author of this code found this surprising, but
  // there's likely a second-order prefetching going on that causes some false
  // sharing.
  struct alignas(CACHELINE_ALIGNMENT * 2) insertion_db_vector
      : std::vector<std::vector<ZT>> {

    size_t bytes_used() const noexcept {
      size_t val{};
      for (const auto &v : *this) {
        val += v.capacity() * sizeof(ZT);
      }

      return val + this->capacity() * sizeof(std::vector<ZT>);
    }
  };

  std::vector<insertion_db_vector> insertion_dbs;
  bgj1_bucketing_interface bgj1_buckets;

  MPI_Comm stop_comm;
  MPI_Datatype vector_type;
  MPI_Datatype entry_type;
  MPI_Datatype sync_header_type;

  alignas(2 * CACHELINE_ALIGNMENT) std::atomic<unsigned> sat_drop;

  unsigned bucket_batches;
  unsigned scratch_buffers;

  std::vector<int> finished_batches;
  std::vector<uint8_t> finished_sizes;

  std::vector<int> scounts, sdispls, rcounts, rdispls;

  struct alignas(2 * CACHELINE_ALIGNMENT) padded_atom_unsigned
      : std::atomic<unsigned> {};

  struct alignas(2 * CACHELINE_ALIGNMENT) padded_atom_int64_t
      : std::atomic<int64_t> {};

  padded_atom_unsigned free_scratch;
  std::vector<padded_atom_unsigned> scratch_used;

  // This can be non-atomic: we can pass this as a parameter to the sieving
  // routines directly.
  std::vector<unsigned> scratch_lookup;

  struct alignas(2 * CACHELINE_ALIGNMENT) scratch_vector {
    std::vector<ZT> incoming_scratch;
    std::vector<ZT> outgoing_scratch;

    size_t bytes_used() const noexcept {
      return sizeof(ZT) *
             (outgoing_scratch.capacity() + incoming_scratch.capacity());
    }
  };

  alignas(2 * CACHELINE_ALIGNMENT) std::vector<scratch_vector> scratch_space;

  int size_for_header;
  uint64_t extra_memory;

  unsigned nr_real_buckets;

  // BDGL stuff.
  std::vector<uint32_t> bdgl_owned_buckets;
  std::vector<std::vector<uint32_t>> bdgl_bucket_map;
  std::vector<uint32_t> bdgl_used_buckets;
  std::vector<uint64_t> sizes;
  std::vector<char> completed;
  std::vector<LFT> bdgl_len;

  struct bdgl_batch {
    unsigned pos;
    unsigned size;
  };

  std::vector<bdgl_batch> bdgl_batch_info;

  unsigned bdgl_nr_completed;
  unsigned bdgl_nr_outstanding;

  padded_atom_unsigned nr_remaining_buckets;

  std::vector<padded_atom_int64_t> bdgl_insert_pos;

  size_t bdgl_nr_buckets, bdgl_bucket_size;
  unsigned bdgl_nr_insertions;

  struct alignas(2 * CACHELINE_ALIGNMENT) UidManageStruct {
    std::mutex lock;
    std::vector<UidType> buffer;
  };

  std::vector<UidManageStruct> to_remove;
  std::vector<padded_atom_unsigned> bdgl_bucket_active;

#ifdef MPI_TIME
  // Marked as mutable to allow calling from const contexts.
  mutable MPITimer timer{};

public:
  std::atomic<uint64_t> recomp_time;

  struct RecompTimer {
    std::chrono::time_point<std::chrono::steady_clock> time;
    MPIObj *root;
    ~RecompTimer() {
      root->recomp_time.fetch_add(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - time)
              .count());
    }
  };

  inline RecompTimer time_recomp();

private:
#define TIME(state) const auto _time = timer.time(state);
#define TIME_RECOMP(mpi) const auto _val = mpi.time_recomp();
#else
public:
  inline void time_recomp();

private:
#define TIME(state)
#define TIME_RECOMP(mpi)
#endif

  /**
     slot_map. This array contains the mapping of UID indices to nodes.
  Concretely, slot_map[i] contains the rank of the node on the network
  resposible for insertions at index i. This is set up once during construction
  and then never modified.
  **/
  alignas(2 * CACHELINE_ALIGNMENT) std::array<int, DB_UID_SPLIT> slot_map;

  static constexpr auto sub_line =
      "dimension,unique sends,nr sends,center bandwidth(b), "
      "bucket bandwidth(b),full bandwidth(b),messages,cpu time(s), "
      "max memory used(b),estimated database size(b),bgj1 times(ms),bdgl "
      "times(ms),cc "
      "times(ms)";

#if defined(MPI_TRACK_UNIQUES) || defined(MPI_TRACK_BANDWIDTH) ||              \
    defined(MPI_TIME)
  std::string filepath_for_track;
#endif

#ifdef MPI_TRACK_UNIQUES
  uint64_t unique_sends;
  uint64_t nr_sends;

  std::atomic<uint64_t> bucket_size;
  std::atomic<uint64_t> unique_entries;
#endif
#endif

public:
  /**
     MPIObj. This constructor creates a new MPIObj. This function will
  initialise the MPI runtime environment if it does not already exist.
  This function may throw in debug builds, but does not throw in release
  builds.

     @param[in] is_g6k: true if the rank is a root rank (i.e running G6K's
  Cython layer), false otherwise.
     @param[in] in_comm: the communicator to use to bootstrap communications.
  This is a type cast of an MPI communicator. This is mostly used during
  testing: a default value of `0` implies MPI_COMM_WORLD (the default
  communicator).
  @param[in] my_threads: the number of threads associated with this MPI_obj.
  Default of 1. Note that in debug builds we throw if this value is 0.
     @param[in] type: the type of topology to construct. Default is a Shuffle
  (i.e Ring) siever.
  @remarks Note that this constructor never frees `in_comm`, even if `in_comm`
  isn't used by this object. It is the responsibility of the caller to handle
  this.
  **/
  MPIObj(const bool is_g6k, const uint64_t in_comm,
         const uint64_t my_threads = 1,
         DistSieverType type = DistSieverType::ShuffleSiever,
         unsigned scale_factor = 1, const unsigned bucket_batches = 1,
         const unsigned scratch_buffers = 1) MPI_DIST_MAY_THROW;
#ifdef G6K_MPI
  /**
     MPIObj. This constructor creates a new MPIObj. This function will
     initialise the MPI runtime environment if it does not already exist.
         @param[in] is_g6k: true if the rank is a root rank (i.e running G6K's
  Cython layer), false otherwise.
     @param[in] in_comm: the communicator to use to bootstrap communications.
  This is mostly used during testing: a default value of `0` implies
  MPI_COMM_WORLD (the default communicator).
     @param[in] my_threads: the number of threads associated with this MPI_obj.
  Default of 1.
     @param[in] type: the type of topology to construct. Default is a Shuffle
  (i.e Ring) siever.
  @remarks Note that this constructor never frees `in_comm` even if `in_comm`
  isn't used by this object. It is the responsibility of the caller to handle
  this.
  **/
  MPIObj(const bool is_g6k, const MPI_Comm in_comm,
         const uint64_t my_threads = 1,
         DistSieverType type = DistSieverType::ShuffleSiever,
         unsigned scale_factor = 1, const unsigned bucket_batches = 1,
         const unsigned scratc_buffers = 1) MPI_DIST_MAY_THROW;

  /**
     ~MPIObj.
  **/
  ~MPIObj() noexcept;
#else
  /**
     MPI.
  **/
  MPIObj() noexcept;
#endif

#ifdef G6K_MPI
  /**
     set_comm. This sets the communicator of `this` MPIObj to `comm_`. This
     function should be called after a new topology was created. Note that this
  is already called by the default constructor, so this should only be used if
  the topology changes at runtime for some reason. This function does not
  throw.
     @param[in] comm_ the communicator to set.
  **/
  void set_comm(MPI_Comm comm_) noexcept;
  /**
     get_comm. This retrieves a copy of the communicator held by `this` MPIObj.
   This function does not throw and simply returns a copy.
     @returns a copy of the communicator held by this MPIObj.
   **/
  MPI_Comm get_comm() noexcept;

  /**
     get_buckets_per_rank. This is a getter for the number of buckets per rank.
     This function returns a copy of the number of buckets per rank. This
  function does not throw.
     @return a copy of the number of buckets per rank.
  **/
  inline uint64_t get_buckets_per_rank() noexcept;
  /**
     get_memory_per_rank. This is a getter for the memory per rank.
     This function returns a copy of the memory per rank. This function
     does not throw.
     @return a copy of the memory per rank.
  **/
  inline std::vector<uint64_t> get_memory_per_rank() noexcept;

  /**
     get_bucket_position. This function retrieves the receiving position for
  centers for rank "i". This function is primarily useful by e.g external
  callers who may need to initialise read data. This function never throws.
     @param[in] o_rank: the rank whose bucket_position is needed. Must not be
  the same as get_rank().
  @param[in] batch: the batch that is being retrieved.
  @return the receiving position for rank "o_rank".
  **/
  inline uint64_t get_bucket_position(const unsigned o_rank,
                                      const unsigned batch) noexcept;

  /**
     get_size_requests. This function returns a reference to the size_requests
  vector. This is mostly useful during testing: see the documentation for
  size_requests to understand why this is useful. This function does not throw.
  Note that the reference returned here is non-const.
     @return a reference to the size_requests vector.
  **/
  std::vector<MPI_Request> &get_size_requests() noexcept;

  inline uint64_t get_bucket_size(const unsigned rank) noexcept;

  /**
   get_outgoing_size_requests. This function returns a reference to the
  outgoing_size_requests vector. This is mostly useful during testing: see the
  documentation for outgoing_size_requests to understand why this is useful.
  This function does not throw. Note that the reference returned here is
  non-const.
   @return a reference to the outgoing_size_reqs vector.
  **/
  std::vector<MPI_Request> &get_outgoing_size_requests() noexcept;

#endif

  /**
     get_topology. This function returns the type of topology attached to this
  particular MPI object. This function does not throw. This function returns
  the null topology type if G6K_MPI is not set.
     @return the type of siever.
  **/
  inline G6K_MPI_CONDITIONAL_CONSTEXPR DistSieverType
  get_topology() const noexcept;

  /**
     is_root. This function returns true if this rank is the root rank and
  false otherwise. This function never throws. Note that this function always
  returns false if MPI is not enabled.
     @return true if this rank is the root rank.
  **/
  inline G6K_MPI_CONDITIONAL_CONSTEXPR bool is_root() const noexcept;

  /**
     is_sieving. This function returns true if this MPI object has been marked
  as running a sieving operation. This function exists solely to prevent
  certain messages being sent during sieving operations (e.g EL). This function
  never throws. Note that this function always returns false if MPI is not
  enabled.
     @return true if this object is in a sieving state, false otherwise.
  **/
  inline G6K_MPI_CONDITIONAL_CONSTEXPR bool is_sieving() const noexcept;

  /**
     is_in_context_change. This function returns true if this MPI object has
   been marked as running a context change operation. This function exists
   solely to prevent extra intialize_local calls from being sent across the
   network. Note that this function always returns false if MPI is not enabled.
     @return true if this object is in a context change, false otherwise.
   **/
  inline G6K_MPI_CONDITIONAL_CONSTEXPR bool
  is_in_context_change() const noexcept;

  /**
     in_sieving. This function marks this MPI object as being in a sieving
   stage if MPI is enabled and nothing otherwise. This prevents extra
   parallel_sort operations from being broadcast across the network. This
   function never throws.
   **/
  inline void in_sieving() noexcept;

  /**
   out_of_sieving. This function marks this MPI object as no longer
   sieving if MPI is enabled and does nothing otherwise. This
   function does not throw.
  **/
  inline void out_of_sieving() noexcept;

  /**
     in_context_change. This function marks this MPI object as being in a
  context change if MPI is enabled and does nothing otherwise..
  This prevents extra initialize_local messages from being
  broadcast across the network. This function never throws.
  **/
  inline void in_context_change() noexcept;

  /**
     out_of_context_change. This function marks this MPI object as no longer in
     a context change if MPI is enabled and does nothing otherwise. This
  function does not throw.
  **/
  inline void out_of_context_change() noexcept;

  /**
     get_root_rank. This function returns the root rank for this communicator.
     This function will always 0 if G6K_MPI is not set. Otherwise, this
  function will always return MPIWrapper::global_root_rank. This function does
  not throw.
     @return the root rank of this rank.
  **/
  int get_root_rank() const noexcept;

  /**
     get_rank. This function returns the rank for this mpi object in the
  current communicator. This function will always return 0 if G6K_MPI is not
  set. This function does not throw.
     @return the rank of this rank.
  **/
  inline G6K_MPI_CONDITIONAL_CONSTEXPR int get_rank() const noexcept;

  /**
     is_distributed_sieving_enabled. This function returns true if
     G6K_MPI is set and false otherwise. This function does not throw.
     @return true if distributed sieving is enabled.
  **/
  static inline constexpr bool is_distributed_sieving_enabled() noexcept;

  /**
     is_active. This function returns true if we are actively sieving using MPI
  and false otherwise. This corresponds to a database that is split across
  multiple ranks. This function does not throw.
     @return true if we are currently sieving using MPI.
  **/
  inline G6K_MPI_CONDITIONAL_CONSTEXPR bool is_active() const noexcept;

  /**
     should_sieve. This function returns true if we should be distributed
  sieving and false otherwise. This corresponds to checking if the current
  dimension `n` is greater than or equal to the threshold. This function does
  not throw.

  @param[in] n: the current sieving dimension.
  @param[in] threshold: the dimension we should start sieving in.
  @return true if we should distributed sieve, false otherwise.
  **/
  inline G6K_MPI_CONDITIONAL_CONSTEXPR bool
  should_sieve(const unsigned n, const unsigned threshold) const noexcept;

  /**
     send_gso. This function broadcasts the full GSO object from the `root
rank` to all other ranks using `comm`. This corresponds to sendiing the GSO
object from the rank that is running the Cython layer associated with G6K to
all other ranks. Note that calling this function with a non-root rank is
undefined behaviour. Note that this function never throws in release builds,
but it may in test builds.

@param[in] full_n: the dimension of the mu object. The receiver must
square-root this to get the full dimension of the lattice.
@param[in] mu: the GSO object. Must not be null.

   **/
  void send_gso(unsigned int full_n,
                const double *const mu) const MPI_DIST_MAY_THROW;

  /**
       receive_gso. This function receives a GSO object from the rank
  `root_rank` (on the `comm`) and overwrites the parameter `gso`. This function
  will first read the number of entries needed and then read the GSO object.
  This function never in release builds, but it may throw in test builds. Note
  that calling this function with a non-root rank is undefined behaviour.

  @param[out] gso: the gso object to overwrite.
  **/
  void receive_gso(std::vector<double> &gso) const MPI_DIST_MAY_THROW;

  /**
    send_status. This function sends the siever's status from the `root` rank
  to all other ranks on `comm`. This function essentially just does a
  broadcast. This function does not throw during release builds, but may in
  test builds. Note that calling this function with a non-root rank is
  undefined behaviour.
    @param[in] status: the status to send.
  **/
  void send_status(const unsigned status) const MPI_DIST_MAY_THROW;

  /**
     broadcast_params. This function broadcasts parameters from the root rank
     to all other ranks in the cluster.
       This function does not throw during release builds, but may in test
  builds. Note that calling this function with a non-root rank is undefined
  behaviour.
    @param[in] params: the parameters to send.
    @param[in] seed: the seed of the rng.
  **/
  void broadcast_params(const SieverParams &params,
                        const unsigned long seed) const MPI_DIST_MAY_THROW;

  /**
     receive_params. This function receives parameters from the root rank
     and returns the result in `params`. This function does not throw during
  release builds but may in test builds. Note that calling this function with a
  root rank is undefined behaviour.
     @param[out] params: a reference to the params object to overwrite.
     @param[out] seed: the location to store the received seed for the rng.
  **/
  void receive_params(SieverParams &params,
                      uint64_t &seed) const MPI_DIST_MAY_THROW;

  /**
   receive_initial_setup. This function reads the starting sieving parameters
   from the root rank and the gso object, storing the results in `params` and
   `gso` respectively. This function essentially merges receive_params and
   receive_gso to make life easier for callers. This function may throw in
debug builds. Note that it is undefined behaviour to call this function with a
root rank.
   @param[out] params: the location to store the received params.
   @param[out] gso: the location to store the received gso object.
   @param[out] seed: the location to store the received seed for the rng.
  **/
  void receive_initial_setup(SieverParams &p, std::vector<double> &gso,
                             uint64_t &seed) const MPI_DIST_MAY_THROW;

  /**
     receive_gso_no_header. This function reads the GSO object from the root
  rank and returns the result in `gso`. This function does not throw during
  release builds but may in test builds. Note that calling this function with a
  root rank is undefined behaviour.
  @param[in] size: the size of the gso to read.
  @param[out] gso: the location to write the read gso object.
  **/
  void receive_gso_no_header(const unsigned size,
                             std::vector<double> &gso) const MPI_DIST_MAY_THROW;

  /**
     receive_gso_update_postprocessing. This function reads the change of basis
     matrix `M` from the root rank and returns the result in `M`. This function
     also returns the (l_, r_, should_redist) values from the root. This
  function does not throw during release builds but may in test builds. Note
  that calling this function with a root rank is undefined behaviour.
     @param[out] M: the location to write the change of basis matrix.
     @param[in] old_n: the old value of n.
     @return (l_, r_, should_redist).
  **/
  std::array<unsigned, 3> receive_gso_update_postprocessing(
      std::vector<long> &M, const unsigned old_n) const MPI_DIST_MAY_THROW;

  /**
     broadcast_el. This function broadcasts the extend left operation to all
  attached ranks. This function does not throw in release builds, but may throw
  during tests. Note that calling this function with a non-root rank is
  undefined behaviour.
     @param[in] lp: the extend left parameter.
  **/
  void broadcast_el(const unsigned lp) const MPI_DIST_MAY_THROW;

  /**
     broadcast_er. This function broadcasts the extend right operation to all
  attached ranks. This function does not throw in release builds, but may throw
  during tests. Note that calling this function with a non-root rank is
  undefined behaviour.
     @param[in] rp: the extend right parameter.
  **/
  void broadcast_er(const unsigned rp) const MPI_DIST_MAY_THROW;

  /**
   broadcast_sl. This function broadcasts the shrink left operation to all
attached ranks. This function does not throw in release builds, but may throw
during tests. Note that calling this function with a non-root rank is undefined
behaviour.
   @param[in] lp: the shrink left parameter.
**/
  void broadcast_sl(const unsigned lp,
                    const bool down_sieve) const MPI_DIST_MAY_THROW;

  /**
     receive_context_change. This function receives a context change message
  from the root rank. This function may throw during testing, but does not
  throw in release builds. Note that calling this function with a root rank is
  undefined behaviour.
  @return a context change header in position 0, with potentially useful data
  at position 1.
  **/
  std::array<unsigned, 2> receive_command() const MPI_DIST_MAY_THROW;

  /**
     receive_best_lifts_as_root. This function accepts a vector of LiftEntries
  and returns the set of short lifts globally. This function must be
  called by the root rank: otherwise, the behaviour is undefined. This function
  may throw during tests, but does not throw in release builds.
     @param[in] lifts_in: the set of best lifts for this rank.
     @param[in] full_n: the dimension of each best lift.
     @return the set of globally shortest lifts.
  **/
  std::vector<LiftEntry>
  receive_best_lifts_as_root(std::vector<LiftEntry> &lifts_in,
                             const unsigned full_n) const MPI_DIST_MAY_THROW;

  /**
     share_best_lifts_with_root. This function accepts a vector of LiftEntries
  and globally computes the set of shortest list entries globally. This
  function must be called by a non-root rank: otherwise, the behaviour is
  undefined. This function may throw during tests, but does not throw in
  release builds.
     @param[in] lifts_in: the set of lift entries from this rank.
     @param[in] full_n: the dimension of each best lift.
  **/
  void
  share_best_lifts_with_root(const std::vector<LiftEntry> &lifts_in,
                             const unsigned full_n) const MPI_DIST_MAY_THROW;

  /**
     receive_whole_database. This function accepts a vector of Entries and
     inserts all remaining vectors into it. This function is only executed when
     the siever is no longer doing distributed sieving. This function resizes
   the `db` parameter (leaving existing entries in place) and inserts the new
   vectors into the latter portion of the database.

     Note that this function must be called by a root-rank: otherwise, the
   behaviour is undefined. This function may throw during tests, but does not
   throw in release builds.
     @params[in] db: the location to hold the new vectors.
   **/
  void receive_whole_database(std::vector<Entry> &db) const MPI_DIST_MAY_THROW;

  /**
     split_database. This function splits the database `db` across all attached
  ranks. The internals of this particular function depend on the topology that
  is used. However, at a high-level we attempt to split the database roughly
  depending on the memory per rank that was provided at the beginning of the
  program.
  **/
  void split_database(std::vector<Entry> &db) const MPI_DIST_MAY_THROW;

  /**
     setup_database. This function either splits the database globally (if
  called by a root rank) or collects the database segments destined for this
  rank (if called by a non-root rank). This function does not throw.
     @param[in] cdb: the cdb.
     @param[in] db: the db.
     @param[in] n: the number of elements per vector.
     @remarks Note that no invariants are set-up by this function: the caller
  must do all initialisation.
  **/
  void setup_database(std::vector<CompressedEntry> &cdb, std::vector<Entry> &db,
                      const unsigned n) noexcept;

  /**
     reconcile_database. This function globally reconciles the database at the
  root rank. Concretely, if this function is called by the root rank, then when
  this function returns the `db` argument will contain the entire database in
  the sieve. By contrast, if this function is called by a non-root rank, then
  when this function returns the `cdb` and `db` will be empty. Note that this
  function also marks this MPI object as inactive.
     @param cdb: the compressed database. This is set to size 0 if the caller
  is a non-root rank. If the caller is a root rank, then all new cdb entries
  have the property of cdb[i].i = i (e.g there is a direct 1:1 correspondence
  betwene entries in cdb and db).
     @param db: the database. This is set to size 0 if the caller is a non-root
  rank. If the caller is a root rank, then new vectors are appended to the end
  of db. Note that the new db values are not initialised.
     @param[in] n: the dimension of each reconcilled vector.
  **/
  void reconcile_database(std::vector<CompressedEntry> &cdb,
                          std::vector<Entry> &db, const unsigned n) noexcept;

  /**
     get_cdb_size. This function either gathers the global cdb size (if
  distributed sieving is enabled) or returns the `in_db` parameter. This is
  primarily to make certain definitions easier to write in other files.
     @param[in] in_db: the size of the rank's cdb.
     @returns the size of the database globally.
  **/
  size_t get_cdb_size(const size_t in_db) const noexcept;

  /**
     get_global_saturation. This function either gathers the global
  saturation count (if distributed sieving is enabled) or returns the
  `in_sat` parameter. This is primarily to make certain definitions easier
  to write in other files.
     @param[in] in_sat: the saturation count for this rank.
     @return the saturation count globally.
  **/
  size_t get_global_saturation(const size_t in_sat) const noexcept;

  /**
     number_of_ranks. This function returns the number of ranks in `comm`.
     This function does not throw.
     @return the number of ranks in the communicator.
  **/
  inline int number_of_ranks() const noexcept;

  /**
     get_total_buckets. This function returns the total number of buckets on
  the network. This is primarily for accurate trial counters across all ranks.
  Please note that this function runs (at present) in O(1) for caching reasons.
  This function does not throw. If MPI is not enabled this function always
  returns 0.
     @return the number of buckets on the network.
  **/
  inline G6K_MPI_CONDITIONAL_CONSTEXPR uint64_t
  get_total_buckets() const noexcept;

  void forward_and_gather_all(
      const unsigned n,
      const std::vector<std::vector<std::vector<std::vector<unsigned>>>>
          &cbuckets,
      const std::vector<Entry> &db) noexcept;

  void forward_and_gather_all(const unsigned n,
                              const std::vector<uint32_t> &buckets,
                              const std::vector<CompressedEntry> &cdb,
                              const std::vector<Entry> &db) noexcept;

  void bdgl_gather_sizes(const std::vector<uint64_t> &sizes) noexcept;
  void bdgl_batch_received(const unsigned batch) noexcept;
  inline bool has_bdgl_finished_sieving() const noexcept;
  inline void bdgl_pass_stop() noexcept;
  inline void bdgl_decrement_outstanding_batches(const unsigned nr) noexcept;

  inline bool bdgl_has_finished() const noexcept;
  inline bool bdgl_has_finished_contributing() const noexcept;

  void bdgl_bucket_received(const unsigned pos) noexcept;
  void bdgl_run_queries(UidHashTable &hash_table,
                        thread_pool::thread_pool &pool) noexcept;

  inline void bdgl_clear_thread_entry(const unsigned id) noexcept;

  inline void mark_bdgl_bucket_as_finished() noexcept;
  unsigned add_to_outgoing_bdgl(const Entry &e, const unsigned index,
                                const unsigned n) noexcept;

  void bdgl_remove_speculatives(UidHashTable &hash_table) noexcept;
  size_t get_bdgl_queue_size() noexcept;

  void bdgl_extract_entry(const unsigned thread_id, Entry &e,
                          const unsigned index, const unsigned n) noexcept;

private:
  /**
     forward_and_gather_buckets. This function issues an MPI_AlltoAllv to
  gather and distribute all vectors relating to `batch`. This function uses
  the elements collected in `serialisation_entries` during bucketing for
  this.
     @param[in] n: the number of elements in each lattice vector.
     @param[in] batch: the batch that is being operated on.
     @param[in] cbuckets: the buckets to send.
     @param[in] db: the database from which entries are drawn.
  **/
  void forward_and_gather_buckets(
      const unsigned n, const int batch,
      const std::vector<std::vector<std::vector<unsigned>>> &cbuckets,
      const std::vector<Entry> &db) noexcept;

  void forward_and_gather_buckets_bdgl(const unsigned n, const int batch,
                                       const std::vector<uint32_t> &buckets,
                                       const std::vector<CompressedEntry> &cdb,
                                       const std::vector<Entry> &db) noexcept;

public:
  /**
     deal_with_finished. This function unpacks the bucket (batch, index) into
     `bucket`. This function should only be called by a thread that is
  sieving.
     @param[in] batch: the batch to which the bucket belongs.
     @param[in] n: the number of ZTs in each vector.
     @param[in] index: the ID for the particular bucket.
     @param[in] size: the number of vectors held on this node already for the
  bucket (batch, index).
     @param[in] scratch_index: the index of the scratch buffer to use.
     @param[in] bucket: the location to store the unpacked vectors.
  **/
  void deal_with_finished(const unsigned batch, const unsigned n,
                          const unsigned index, const unsigned size,
                          const unsigned scratch_index,
                          std::vector<Entry> &bucket) noexcept;

  /**
     db_size. This function returns the size of the global database is MPI
     is enabled or `in_size` otherwise. This function does not throw. This
  function differs from get_cdb_size because it prepends a header to tell any
  attached ranks to return their DB sizes: get_cdb_size does not prepend this
  header.
     @param[in] in_size: the size of the database held by this rank.
     @return the size of the database globally.
  **/
  size_t db_size(const size_t in_size) const noexcept;

  size_t db_capacity(const size_t capacity) const noexcept;

  /**
     grow_db. This function grows the database globally to contain
     at least `N` vectors. These vectors
  are guaranteed to be unique. This function does not throw in
  release builds but may throw in debug builds. If MPI
  is not enabled then this function does nothing: however, this function should
  not really be called unless MPI is enabled. Please note that it is undefined
  behaviour to call this function with a non-root rank.
     @param[in] N: the size of the resized database.
     @param[in] large: the large parameter from G6K. See the documentation from
  Siever::sample for more.
     @param[in] siever: the siever object to resize.
     @return true if _this_ call to grow_db grew the database, false otherwise.
  **/
  bool grow_db(const size_t N, const unsigned large,
               Siever &siever) const MPI_DIST_MAY_THROW;

  /**
     reserve_db. This function allocates more memory globally, so that there is
  storage for at least `N` vectors. This function does not throw in
  release builds but may throw in debug builds. If MPI
  is not enabled then this function does nothing: however, this function should
  not really be called unless MPI is enabled.
     @param[in] N: the size of the resized database.
     @param[in] siever: the siever object to resize.
     @return true if _this_ call to reserve_db grew the database, false
  otherwise.
  **/

  bool reserve_db(const size_t N, Siever &siever) const MPI_DIST_MAY_THROW;

  /**
     shrink_db. This function shrinks the database to contain `N` vectors.
     This function attempts to retain the best `N` vectors globally using an
  iterative search method (see MPIWrapper::shrink_db for more). If MPI is not
  enabled then this function does nothing. This function does not throw in
  release builds but may throw in debug builds. Please note that it is undefined
  behaviour to call this function with a non-root rank.
     @param[in] N: the size of the resized database.
     @param[in] siever: the siever object to resize.
     @return true if _this_ call to shrink shrank the database, false
  otherwise.
  **/
  bool shrink_db(const size_t N, Siever &siever) const MPI_DIST_MAY_THROW;

  /**
     sort_db. This function sends a "sort db" message to all attached ranks.
     This has the effect of locally sorting each cdb. This function may throw
     in debug builds. Note that calling this function with a non-root rank is
     undefined behaviour.
  **/
  void sort_db() const MPI_DIST_MAY_THROW;

  /**
     gso_update_postprocessing. This function sends a message to all attached
  ranks to get them to run the GSO postprocessing on their database. See the
  documentation of gso_update_postprocessing in siever.h for more. This
  function does not throw in release builds but may throw in debug builds. Note
  that it is undefined behaviour to call this function from a non-root rank.
     @param[in] l_: the new l value.
     @param[in] r_: the new r value.
     @param[in] old_n: the old n value.
     @param[in] should_redist: true if we should call the redistribution
  functionality, false otherwise.
     @param[in] M: the change of basis transformation. This is a C-style 1D
  array representing a 2D matrix. Must be of size (r_ - l_) * old_n.
  **/
  void gso_update_postprocessing(const unsigned int l_, const unsigned int r_,
                                 const unsigned old_n, const bool should_redist,
                                 long const *M) MPI_DIST_MAY_THROW;

  /**
     send_stop. This function sends a stop from a root rank to all attached
  ranks. This tells all attached ranks to terminate their current event loop.
  This function does not throw in release builds but may throw in debug modes.
     Note that this function must not be called by a non-root rank: otherwise,
  the behaviour is undefined.
  **/
  void send_stop() MPI_DIST_MAY_THROW;

  /**
     start_bgj1. This function sends a message to all attached ranks to
  indicate that BGJ1 sieving is to begin. This function must not be called by a
  non-root rank. This function may throw in debug builds but does not throw in
  release.
     @param[in] alpha: the alpha parameter for sieving.
  **/
  void start_bgj1(double alpha) const MPI_DIST_MAY_THROW;

  /**
     receive_alpha. This function receives the value of alpha from the root
  rank. This is the value that is to be used inside BGJ1 sieving. This function
  does not throw in release builds but may throw during debug builds. Note that
  calling this function from a root rank will introduce undefined behaviour.
  **/
  double receive_alpha() const MPI_DIST_MAY_THROW;

  /**
     broadcast_initialize_local. This function broadcasts an initialize local
   message to all attached ranks. This function does not throw in release
   builds but may in debug builds. This function must not be called by a
   non-root rank: otherwise, the behaviour is undefined.
     @param[in] ll_ the ll_ argument to initialize_local.
     @param[in] l the l argument to initialize_local.
     @param[in] r the r argument to initialize_local.
   **/
  void broadcast_initialize_local(const unsigned ll, const unsigned l,
                                  const unsigned r) MPI_DIST_MAY_THROW;

  /**
     receive_l_and_r. This function receives the l and r parameters from an
  initialize_local message sent by the root rank. This function must be called
  by a non-root rank: otherwise the behaviour is undefined. This function does
  not throw in release builds but may in debug builds.
     @return the l and r arguments from initialize_local. l is placed at index
  0, with r at index 1.
  **/
  std::array<unsigned, 2> receive_l_and_r() MPI_DIST_MAY_THROW;

  /**
     build_global_histo. This function builds the histogram of norms needed for
  G6K. The behaviour of this function depends on the caller.

     - If the caller is a root rank (and if need_message is set) then this
  function broadcasts a message to all attached ranks, and then stores the
  histogram produced globally in `histo`. If need_message is not set then there
  is no message broadcast.

     - If the caller is a non-root rank, then this function simply sends the
  histogram in `histo` to the root rank.

     This function does not throw.

     @param[in] histo: a pointer to the histogram produced by G6K. This must be
  pre-populated and be a pointer to a G6K hisotgram object. We throw away type
  safety here to avoid circular dependencies on siever.h.
  **/
  void build_global_histo(long *const histo, const bool need_message) noexcept;

  /**
   gather_gbl_sat_variance. This function computes the population variance of
GBL_sat_count across all attached ranks and returns the result to the root
rank. For all non-root ranks, the function returns 0.0. This function never
throws.
   @param[in] cur_sat: the saturation count for this rank.
   @return the variance for the saturation count across all ranks or 0.0 if the
caller is a non-root rank.
**/
  double gather_gbl_sat_variance(const size_t cur_sat) noexcept;
  /**
 gather_gbl_ml_variance. This function computes the population variance of
GBL_max_len across all attached ranks and returns the result to the root rank.
For all non-root ranks, the function returns 0.0. This function never throws.
 @param[in] cur_gbl_ml: the GBL_max_len for this rank.
 @return the variance for the GBL_max_len across all ranks or 0.0 if the caller
is a non-root rank.
**/
  double gather_gbl_ml_variance(const double cur_gbl_ml) noexcept;

  /**
     setup_bucket_positions. This function constructs the bucket_positions
  vector. For more usage on that vector, see its documentation in this class.
  This function never throws.

     @param[in] size: the size of the sieving database held by G6K.
  process at once.
  **/
  void setup_bucket_positions(const size_t size) noexcept;

  void initialise_thread_entries(const unsigned nr_threads,
                                 const bool is_bdgl = false) noexcept;

  inline ThreadEntry &get_thread_entry(const unsigned index) noexcept;
  inline BdglThreadEntry &get_thread_entry_bdgl(const unsigned index) noexcept;
  inline int64_t get_bdgl_insert_pos(const unsigned index) noexcept;
  inline void bdgl_update_insert_pos(const unsigned index,
                                     const int64_t pos) noexcept;
  void test() noexcept;

  /**
     setup_insertion_requests. This function creates persistent requests for
  both sending and receiving insertion candidates from other nodes on the
  network. Concretely, this function sets up both the incoming and outgoing
  requests, and starts the incoming requests. This function does not throw.
     @param[in] n: the dimension of the vectors that are received.
  **/
  void setup_insertion_requests(const unsigned n,
                                const unsigned lift_bounds_size) noexcept;

  /**
     get_size_and_start. This function returns the number of vectors and the
  starting index of the set of centers for rank `rank`. In particular, this
  function returns the number of centers received and the starting position of
  these centers (relative to G6K's DB) as an array. This function does not
  throw. Note that this function should only be called on values returned by
  has_bucket.
     @param[in] rank: the rank whose centers we are processing.
     @return the number of vectors received and the starting index at indices 0
  and 1 respectively.
  **/
  inline std::array<uint64_t, 2>
  get_size_and_start(const unsigned rank, const unsigned batch) noexcept;

  /**
     collect_statistics. This function collects all sieving statistics at the
  root rank if MPI is enabled and does nothing otherwise. This function never
  throws.
     @param stats: the statistics for this rank. If the caller is a root rank,
  then the global sieving results are stored here.
  **/
  void collect_statistics(SieveStatistics &stats) noexcept;

  /**
     has_received_buckets. This function returns true if we have received all
  outstanding buckets for this process and false otherwise. This function does
  not throw.
     @return true if this process has received all associated buckets, false
  otherwise.
  **/
  bool has_received_buckets() noexcept;

  /**
     inc_sieved_buckets. This function increments the number of sieved buckets.
  This indicates that a particular bucket held by this rank has been processed.
  **/
  void inc_sieved_buckets(const unsigned index) noexcept;

  inline void inc_bucketed_count(const unsigned batch) noexcept;

  inline void bucketing_request_finished(const unsigned batch) noexcept;

  inline MPIRequest::ReqSpan finished_sieving() noexcept;
  inline MPIRequest::ReqSpan finished_insertion_sizes() noexcept;
  inline MPIRequest::ReqSpan finished_incoming_insertions() noexcept;
  inline MPIRequest::ReqSpan finished_insertions() noexcept;
  inline MPIRequest::ReqSpan finished_buckets() noexcept;
  inline MPIRequest::ReqSpan finished_processed_buckets() noexcept;

  /**
     has_finished_sieving_ours. This function returns true if the number of
   sieved buckets is equal to the number of buckets held by this process and
   false otherwise. This function does not throw.
     @return true if we have sieved all buckets, false otherwise.
   **/
  bool has_finished_sieving_ours(const unsigned index) noexcept;

  /**
     reset_sieving_count. This function resets the number of currently sieved
  buckets to 0. This function does not throw.
  **/
  void reset_sieving_count(const unsigned index) noexcept;

  /**
     cancel_outstanding. This function cancels all outstanding MPI requests.
  This is typically only useful when sieving ends to prevent overlapping
  communications. This function also clears various states for this particular
  sieving iteration: thus, it should be called when a sieving iteration
  finishes.
  **/
  void cancel_outstanding() noexcept;

  void finish_bdgl() noexcept;
  /**
     get_nr_buckets. This function returns the number of buckets that are
  processed per iteration by this process. This function does not throw. Note
  that this function will always return the same value per object. This
  function does not modify this object.
     @return the number of buckets processed per iteration by this process.
  **/
  inline size_t get_nr_buckets() const noexcept;

  /**
     reset_stats. This function broadcasts a reset stats call to all attached
   processes on the communicator. This has the effect of resetting the stats
   counters of all nodes. This function does not throw or modify this object.
   Note that it is undefined behaviour to call this function from a non-root
   process.
   **/
  void reset_stats() const noexcept;

  /**
     get_min_max_len. This function globally computes the smallest "max_len"
  aargument and returns the result. This function does not throw or modify this
  object.
     @param[in] max_len: the max len argument for this node.
     @return the smallest max_len argument across the entire communicator.
  **/
  double get_min_max_len(double max_len) const noexcept;

  /**
     owns_uid. This function returns true if this node is responsible for
  insertions to do with `uid` and false otherwise. This function does not throw
  or modify this object.
     @param[in] uid: the uid to query.
     @return true if this node is responsible for inserting `uid`, false
  otherwise.
  **/
  inline bool owns_uid(const UidType uid) const noexcept;

  /**
     redistribute_database. This function globally re-arranges the database.
  Briefly, this function iterates over the `db` and forwards any entries that
  don't belong to this node to the correct home, removing them from the cdb,
  the db, and the hash_table. In exchange, this node receives any vectors from
  elsewhere on the network that belong to this node. This function does not
  throw. This function returns any duplicates in the database that are received
  during this process. Each node must call this function at the end of
  collision checks.
     @param[in] cdb: the cdb for this node.
     @param[in] db: the db for this node.
     @param[in] n: the dimension of received vectors.
     @param[in] hash_table: the hash table. All sent vectors are removed from
  the hash table, and all incoming ones are added.
     @param[out] duplicates: a vector containing all duplicate entries read.
     @param[out] incoming: a vector containing all received entries.
  **/
  void redistribute_database(std::vector<CompressedEntry> &cdb,
                             std::vector<Entry> &db, const unsigned n,
                             UidHashTable &hash_table,
                             std::vector<unsigned> &duplicates,
                             std::vector<unsigned> &incoming);

  /**
     add_to_outgoing. This function adds `e` to the correct outgoing insertion
  buffer. Note that it is undefined behaviour to call this function with an
  entry that should be inserted by this node. This function does not throw.
     @param[in] e: the Entry to insert in the insertion buffer.
  **/
  void add_to_outgoing(const Entry &e, const size_t index,
                       const unsigned n) noexcept;

  /**
     get_insertion_candidates. This function returns the set of insertion
  candidates received from `rank`. This is returned as a non-const reference to
  the buffer. The returned buffer will not change until insertions have
  finished. This function does not throw.
     @param[in] rank: the rank who sent us insertion candidates.
     @return a non-const reference to the insertion candidates.
  **/
  inline std::vector<ZT> &
  get_insertion_candidates(const unsigned rank) noexcept;

  inline void start_insertions(const unsigned batch) noexcept;

  /**
     mark_insertion_as_done. This function marks the insertion request from
  `rank` as finished. In practice, this function should be used to denote that
  this node has finished processing insertions from `rank`, and that we may
  receive further insertions. This function is thread safe. This function does
  not throw.
     @param[in] rank: the rank whose insertions this node has finished.
  **/
  inline void mark_insertion_as_done(const unsigned rank) noexcept;

  void start_batch(const std::vector<Entry> &db,
                   const std::vector<CompressedEntry> &centers,
                   const unsigned batch, const int64_t trial_count,
                   const std::vector<FT> &lift_bounds) noexcept;

  void grab_lift_bounds(std::vector<FT> &lift_bounds,
                        FT &lift_max_bound) noexcept;

  SyncChange process_incoming_syncs(std::vector<FT> &lift_bounds,
                                    std::vector<Entry> &db,
                                    FT &lift_max_bound) noexcept;

  inline bool is_done() noexcept;
  void process_stop() noexcept;
  inline void issue_stop_barrier() noexcept;
  inline bool can_issue_more(const unsigned batch) const noexcept;
  inline void set_sat_drop(const unsigned drop) noexcept;

  inline void mark_sieving_as_finished(const unsigned batch) noexcept;
  inline bool has_stopped() const noexcept;
  inline bool is_initial_barrier_completed() const noexcept;
  inline bool can_issue_barrier() const noexcept;
  inline bool has_barrier_finished() const noexcept;

  inline void start_sieving_bucket(const unsigned index) noexcept;

  inline void abort(const int error) noexcept;
  inline bool can_issue_stop_barrier() const noexcept;
  inline MPIRequest::ReqSpan finished_syncs() noexcept;

  FT get_norm(const size_t N, const Siever &siever) noexcept;
  FT get_norm_bdgl(const size_t N, const Siever &siever) noexcept;

  void batch_insertions(const size_t index, const unsigned batch,
                        const unsigned n) noexcept;

  void send_sizes(
      const unsigned batch,
      const std::vector<std::vector<std::vector<unsigned>>> &cbuckets) noexcept;

  void send_insertion_sizes(const unsigned batch, const unsigned n) noexcept;

  long double get_cpu_time(const std::clock_t time) const noexcept;

  uint64_t get_extra_memory_used() const noexcept;
  void set_extra_memory_used(const uint64_t use) noexcept;

  void reset_bandwidth() noexcept;

  uint64_t get_total_messages() noexcept;
  uint64_t get_total_bandwidth() noexcept;

  uint64_t get_messages_for(const ContextChange type) noexcept;
  uint64_t get_bandwidth_for(const ContextChange type) noexcept;

  TimerPoint time_bgj1() noexcept;
  TimerPoint time_bdgl() noexcept;

  uint64_t get_bgj1_center_bandwidth() noexcept;
  uint64_t get_bgj1_buckets_bandwidth() noexcept;
  uint64_t get_bgj1_messages_used() noexcept;
  uint64_t get_bgj1_bandwidth_used() noexcept;

  inline unsigned number_of_free_scratch_space() const noexcept;
  void decrease_free_scratch(const unsigned val) noexcept;

  inline void dec_batch_use(const unsigned index) noexcept;
  inline void bdgl_dec_bucket_use(const unsigned index) noexcept;

  inline unsigned get_batch_index(const unsigned batch) noexcept;

  std::array<uint64_t, 2> get_unique_ratio() noexcept;

  void write_stats(const unsigned n, const size_t dbs) noexcept;

  uint64_t get_adjust_timings() noexcept;

  void start_insertions(const unsigned batch, const unsigned n) noexcept;

  inline void mark_batch_as_finished(const unsigned batch) noexcept;

  std::vector<unsigned>
  bdgl_exchange(std::vector<Entry> &db, std::vector<CompressedEntry> &cdb,
                const unsigned n, std::vector<size_t> &sizes,
                std::vector<uint32_t> &buckets, const size_t bsize) noexcept;

  void mark_as_replaced(const UidType removed) noexcept;
  void mark_as_unused(const UidType unused) noexcept;

  std::vector<uint32_t> &setup_bdgl_bucketing(const unsigned n,
                                              const size_t nr_buckets_aim,
                                              const size_t bsize,
                                              const size_t cdb_size) noexcept;

  inline size_t get_number_of_bdgl_buckets() const noexcept;

  inline unsigned get_bdgl_bucket(const unsigned index,
                                  const unsigned offset) const noexcept;

  inline size_t get_bdgl_batch_size(const unsigned index) const noexcept;

  inline void bdgl_mark_bucket_as_processed() noexcept;

  void bdgl_remove_duplicates(const std::vector<Entry> &db,
                              std::vector<std::vector<QEntry>> &t_queues,
                              UidHashTable &hash_table);

  void bdgl_remove_inserted(UidHashTable &hash_table,
                            thread_pool::thread_pool &pool);

  void bdgl_broadcast_lsh(const ProductLSH &lsh) noexcept;

  ProductLSH bdgl_build_lsh() noexcept;

  void start_bdgl(const size_t nr_buckets_aim, const size_t blocks,
                  const size_t multi_hash) noexcept;

  std::array<uint64_t, 3> receive_bdgl_params() noexcept;

  void count_uniques(const std::vector<Entry> &bucket) noexcept;
  void print_uniques(const unsigned iter) const noexcept;

  bool bdgl_gbl_uniques(const std::vector<Entry> &ents);
  bool bdgl_inserts_consistent(const std::vector<Entry> &db,
                               UidHashTable &hash_table) noexcept;

private:
#ifdef MPI_TRACK_UNIQUES
  void
  count_uniques(const std::vector<std::vector<std::vector<unsigned>>> &cbuckets,
                const std::vector<Entry> &db) noexcept;

#endif

  uint64_t compute_extra_memory_used() const noexcept;

  /**
     retrieve_offset. This function retrieves the offset index in certain
  arrays (e.g size_requests etc) for index `i`. Concretely, this function
  returns "i" if i < my_rank and i - 1 if "i >= my_rank". This exists
  primarily to save on certain allocations that otherwise might be difficult
  to program in MPI.
     @param[in] i: the rank ID.
     @param[in] my_rank the rank of this object.
     @return the index needed to retrieve data about rank "i".
     @remarks the returned value here is not useful for buckets_per_rank:
  this is kept internal to avoid confusion.
  **/
  static inline unsigned retrieve_offset(const unsigned i,
                                         const unsigned my_rank) noexcept;

  static inline unsigned retrieve_index(const unsigned i,
                                        const unsigned my_rank) noexcept;

  int size_for_headers(const unsigned size) noexcept;

  unsigned get_unused_scratch_buffer(const int batch) noexcept;

  template <bool is_bgj1>
  void dispatch_to_forward_and_gather_buckets(
      const unsigned n, const int batch,
      const std::vector<std::vector<std::vector<unsigned>>> &cbuckets,
      const std::vector<Entry> &db, const std::vector<int> &sizes,
      MPI_Comm communicator) noexcept;

  inline std::vector<bucket_pair> &
  get_bucket_pairs(const unsigned batch) noexcept;

  void init_thread_entries_bgj1(const unsigned nr_threads) noexcept;
  void init_thread_entries_bdgl(const unsigned nr_threads) noexcept;
};

// Inline definitions live here.
#include "mpi_obj.inl"

#endif
