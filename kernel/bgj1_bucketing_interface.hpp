#ifndef INCLUDED_BGJ1_BUCKETING_INTERFACE_HPP
#define INCLUDED_BGJ1_BUCKETING_INTERFACE_HPP

#include "communicator.hpp"
#include "compat.hpp"
#include "packed_sync_struct.hpp"
#include "siever_types.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

/**
   bgj1_bucketing_interface. This class provides a unified API for accessing
state that's related to bucketing in the distributed BGJ1 code. In particular,
this class is meant to hide the difficulties associated with a variable number
of operations per process, whilst also providing opportunities for potential
speedups.

To understand this class, it is helpful to think about how a single bucket is
produced. The root process first sends out a sync message to all other processes
which contains:

1. The lift bounds.
2. The sync header.
3. The center vector itself.

This is done using an MPI communicator. At this stage, each process (including
the root process) produces a local bucket against the center. The non-root
processes then forward the size of the bucket to the root process, who allocates
enough memory to hold the incoming buckets. Finally, each non-local process
forwards their local buckets to the root process for sieving.

Implicitly, this relies upon certain control barriers: each process needs to
know when it has finished producing a local bucket, or when certain requests
have finished. This becomes rather confusing when the bucketing process is
repeated in parallel across many processes, with a variable number of buckets in
flight at any given time.

To solve this issue, this class provides a series of vectors that control
certain pieces of state. Each bucketing request (which can be composed of more
than one center) is executed using its own state, comprised of (amongst others)
its own local communicator, memory space for sync messages, and operation
counting. This separation of state prevents overlapping requests or confusion.
This state is spread across multiple vectors for better concurrency.

The API for this class is rather straightforward. Most functions require a
single parameter, which indicates which batch is being modified. This then
allows this particular class to track these pieces of state. Notably, because we
deal with process-specific details in e.g MPIObj, this class only needs to track
state for batches at a high-level, with lower level details being dealt with
inside MPIObj.

@remarks As an aside: this class only provides an interface for the various
states that are maintained during the execution of the sieve. This class does
_not_ provide an interface for the internal storage needed for actually storing
the produced buckets. This is a deliberate design choice: as outgoing buckets
require different storage than incoming buckets, it makes sense to save storage
and do all of this externally. Thus, the actual bucket memory should be owned
and operated elsewhere.
**/
class bgj1_bucketing_interface {
public:
  /**
     bgj1_bucketing_interface. This is the default constructor. This does
  nothing: instead, this exists solely to allow the MPIObj to run some
  pre-processing before this object is initialised.
  **/
  bgj1_bucketing_interface() noexcept = default;

  /**
     bgj1_bucketing_interface. This constructor allocates all vectors and state
  that are dimension independent (i.e. the data that does not depend on the
  sieving dimension).
     @param[in] buckets_per_rank: the number of buckets per batch.
     @param[in] nr_ranks: the number of ranks.
     @param[in] batches: the number of batches.
     @param[in] comm: the MPI communicator that is used to bootstrap this class.
  **/
  bgj1_bucketing_interface(const uint64_t buckets_per_rank,
                           const unsigned scale_factor, const unsigned rank,
                           MPI_Comm comm) noexcept;

  /**
     get_comm. This function returns a reference to the communicator used for
  bucketing `batch`.
     @param[in] batch: the batch of the target batch.
     @return a reference to the communicator used for bucketing for `batch`.
  **/
  inline Communicator &get_comm(const unsigned batch) noexcept;

  inline bool is_sizing(const unsigned batch) const noexcept;

  inline unsigned get_next_wait_head(const unsigned start) const noexcept;

  inline unsigned get_next_incoming_centres() const noexcept;

  /**
     inc_sieved_buckets. This function increments the number of sieved buckets
  in batch. This function should only be called if the batch is in
  the sieving state.
     @param[in] batch: the target batch.
  **/
  inline void inc_sieved_buckets(const unsigned batch) noexcept;

  /**
     inc_bucketed_count. This function increments the number of produced buckets
     in `batch`. This function should only be called if the batch
     is in the bucketing stage.
     @param[in] batch: the target batch.
  **/
  inline void inc_bucketed_count(const unsigned batch) noexcept;

  /**
     bucket_position. This function returns the bucket position for `rank` in
  `batch`. Specifically, this function returns the position in the db (and cdb)
  where centers from `rank` and `batch` are stored. This function only returns a
  meaningful value if `rank` differs from the rank of the calling process.
     @param[in] rank: the rank of the target process.
     @param[in] batch: the batch of the target batch.
     @param[in] width: the maximum number of batches.
     @return the index of received centers from `rank` for `batch`.
  **/
  inline uint64_t bucket_position(const unsigned rank, const unsigned batch,
                                  const unsigned width) noexcept;

  /**
     setup_bucket_positions. This function sets up the bucket positions (see
  bucket_position for more).
     @param[in] size: the size of the DB excluding the scratch space.
     @param[in] bucket_batches: the maximum number of batches.
     @param[in] buckets_per_rank: the number of buckets per batch for each
  process.
  @param[in] nr_ranks: the number of ranks in the program.
     @param[in] rank: the rank of the calling process.
  **/
  void setup_bucket_positions(const size_t size, const unsigned bucket_batches,
                              const uint64_t buckets_per_rank,
                              const unsigned nr_ranks,
                              const unsigned rank) noexcept;

  /**
     start_incoming_centers. This function marks the communicator associated
  with `batch` as the incoming centers state and increments the number of issued
  calls.
     @param[in] batch: the target batch.
  **/
  inline void start_incoming_centers(const unsigned batch) noexcept;

  /**
     start_incoming_size. This function marks the communicator with `batch` as
  the incoming_size state, resets the number of completed operations to 0, and
  increments the number of issued calls.
     @param[in] batch: the batch of the target batch.
  **/
  inline void start_incoming_size(const unsigned batch) noexcept;

  /**
     setup_sync_objects. This function sets up the sync objects for incoming
  centers. Specifically, this function allocates enough space for all incoming
  lift bounds (`size` entries per batch).

     @param[in] size: the number of characters needed to receive centers from
  all other processes.
  **/
  void setup_sync_objects(const unsigned size) noexcept;

  inline void mark_sieving_as_finished(const unsigned batch) noexcept;

  /**
     get_sizes. This function returns a reference to the sizes vector
     for batch. The returned vector should not be resized.
     @param[in] batch: the target batch.
     @return a reference to the sizes vector for batch.
  **/
  inline std::vector<int> &get_sizes(const unsigned batch) noexcept;

  /**
     get_outgoing_offsets. This function returns a pointer to the offsets for
     sending a sync message (for the current batch).
     @param[in] batch: the batch.
     @param[in] width: the maximum number of batches.
     @return a pointer to the offsets for sending the sync message for `batch`.
   **/
  inline MPI_Aint *get_outgoing_offsets(const unsigned batch,
                                        const unsigned width) noexcept;

  inline std::vector<char> &get_memory_for(const unsigned batch) noexcept;

  /**
     start_sieving_bucket. This function marks batch as a sieving batch.
     This has the effect of resetting the number of operatiosn and marking the
  communicator as in the sieving state. This should only be called if `rank` is
  the same as the rank of the calling process.
     @param[in] batch: the batch.
  **/
  inline void start_sieving_bucket(const unsigned batch) noexcept;

  /**
     has_finished_sieving_ours. This function returns true if the `batch`
  has been sieved. This returns true iff the communicator state is in the
  sieving state and if the number of processed buckets is equal to `buckets`.
  Note that this only ever refers to the current process' view of sieving.
     @param[in] batch: the target batch.
     @param[in] buckets: the number of buckets per batch.
     @return true if the bucket has been sieved, false otherwise.
  **/
  inline bool has_finished_sieving_ours(const unsigned batch,
                                        const unsigned buckets) noexcept;

  inline std::vector<Communicator> &comms() noexcept;

  /**
     reset_comms. This function resets all communicators to the starting state.
  This should only be called once sieving has finished.
  **/
  inline void reset_comms() noexcept;
  /**
     reset_counts. This function resets all counts and states to zero.
     This should only be called once a sieving iteration has finished.
  **/
  inline void reset_counts() noexcept;
  /**
     reset_ops. This function resets all operation counts to zero. This
     should only be called once a sieving iteration has finished.
  **/
  inline void reset_ops() noexcept;

  /**
     free_comms. This function frees all communicators. This should only be
  called once the sieve has terminated.
  **/
  inline void free_comms() noexcept;

  /**
     can_issue_more. This function returns true if the calling process can
  process more batches for batch. This is true if the stop count and the
  iteration count are not equal.
     @param[in] batch: the target batch.
     @return true if another batch can be processed, false otherwise.
  **/
  inline bool can_issue_more(const unsigned batch) const noexcept;

  /**
     bucketing_request_finished. This function marks the bucketing batch `batch`
     as finished. This means that the calling process has finished
     sending buckets for `batch`. Specifically, this function resets the
     operations count and marks the communicator as in a sieving state.
     @param[in] batch: the target batch.
  **/
  inline void bucketing_request_finished(const unsigned batch) noexcept;

  /**
     states. This function returns a reference to the states vector. Briefly,
  states[i] contains the maximum number of iterations issued globally for a
  particular batch. This data is collected at the end of the sieving iteration
  and used to ensure global consistency.
     @return a reference to the states vector.
  **/
  inline std::vector<uint8_t> &states() noexcept;

  /**
     states. This function returns a reference to the counts vector. Briefly,
  counts[i] contains the number of iterations issued locally for a particular
  batch. This data is used to ensure global consistency.
     @return a reference to the states vector.
  **/
  inline std::vector<uint8_t> &counts() noexcept;

  /**
     has_finished_bucketing. This function returns true if the bucketing for
     batch has been completed. This function returns true if the number of
     finished buckets is equal to the number of ranks.
     @param[in] batch: the batch.
     @param[in] nr_ranks: the number of ranks in the communicator.
     @return true if the batch has been bucketed, false otherwise.
  **/
  inline bool has_finished_bucketing(const unsigned batch,
                                     const unsigned nr_ranks) noexcept;

  /**
     start_incoming_buckets. This function marks the batch as
  receiving incoming buckets. In practice, this function
  resets the number of operations and marks the
  communicator as receiving buckets.
     @param[in] batch: the target batch.
  **/
  inline void start_incoming_buckets(const unsigned batch) noexcept;

  /**
     are_all_done. This function returns true if all outstanding batches for
  this rank are completed. This function returns true iff all communicators are
  in the sieving state and if all states and counts for those batches match.
  This is used to decide if a sieving barrier should be issued. Please note that
  this should only be called if we are at the end of a sieving iteration:
  otherwise, results are not guaranteed.
     @return true if all batches are finished, false otherwise.
  **/
  inline bool are_all_done() const noexcept;

  inline uint64_t bytes_used() const noexcept;

  inline bool is_sieving(const unsigned batch) const noexcept;
  inline bool is_receiving_buckets(const unsigned batch) const noexcept;
  inline unsigned get_next_sizes() const noexcept;

  inline void mark_insertion_as_done(const unsigned batch) noexcept;

  inline bool is_receiving_insertion_sizes(const unsigned batch) const noexcept;
  inline bool has_finished_insertion(const unsigned batch) const noexcept;

  inline void start_incoming_insertion_size(const unsigned batch) noexcept;
  inline void start_incoming_insertion(const unsigned batch) noexcept;

  inline void mark_batch_as_finished(const unsigned batch) noexcept;

  inline void start_insertions(const unsigned batch) noexcept;

  inline bool is_finished(const unsigned batch) const noexcept;

private:
  // These all follow the same regular access pattern.
  std::vector<Communicator> comms_;

  struct alignas(2 * CACHELINE_ALIGNMENT) padded_atom_unsigned
      : public std::atomic<unsigned> {};

  std::vector<padded_atom_unsigned> ops_;
  std::vector<uint8_t> states_;
  std::vector<uint8_t> counts_;

  MPI_Datatype sync_type_;

  std::vector<std::vector<char>> sync_headers_;

  std::vector<std::vector<int>> sizes_;
  std::vector<uint64_t> bucket_positions_;
};

// All inline functions live here.
#include "bgj1_bucketing_interface.inl"

#endif
