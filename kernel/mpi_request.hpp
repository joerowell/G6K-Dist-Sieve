#ifndef INCLUDED_MPI_REQUEST_HPP
#define INCLUDED_MPI_REQUEST_HPP

#include "mpi.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
/**
   MPIRequest. This class exists to make collating multiple MPI requests into
one location easier. Briefly, profiling indicates that storing multiple MPI
requests separately is highly expensive relative to storing them contiguously.
This is because (at least on OpenMPI) testing MPI requests issues a memory
barrier (an MFENCE instruction on x86-64) which globally forces all read and
writes done by other cores (even non-atomic ones!) to be visible to the testing
thread. In a memory intensive application such as sieving this is unbelievably
expensive (i.e. it accounts for around 50% of core time). Moreover, in a single
node application the cost of this fence is even higher than the actual transfer
of data between processes (depending on how well the latency is masked by
multiple buckets being in flight).

   To lower this cost, we amortise the MFENCE across all of the requests that we
are testing. This requires some careful handling: this class is deliberately
designed to be opaque, but it requires careful usage (see MPIObj::test for more
details).

Please note that most of the methods in this class are deliberately inline. This
is because most of the useful functionality is essentially comprised of a series
of getters. The main cost of this class will be calling the test function.

**/

// clang-format off

/**
How this class works:
---------------------------

This class lays out all MPI requests in a single, contiguous buffer. To keep things simple, we
assume that the request order is fixed.

In practice, the number of requests in each portion of the overall array can vary due to runtime parameters.
Thus, it's typically more useful to view the array as being composed of several sub-spans, which look like
this when combined:

+--------------+--------------+--------------+--------------+--------------+--------------+
|  BUCKET OUT  | SIZES        | OUT INSERT   | IN INSERT    | BUCKET IN    | STOP         |
+--------------+--------------+--------------+--------------+--------------+--------------+

Where:

1) BUCKET OUT contains all of the outgoing bucketing requests. These are used for synchronising and sending centers
to other processes in a batch. In practice BUCKET_OUT[i] contains the request for the i-th BATCH.
2) SIZES contains all of the size requests. These are used for gathering all bucketing sizes globally. Similarly to above,
   SIZES[i] contains the size request for the i-th BATCH.
3) OUT_INSERT contains the outgoing insertion requests. These are the batched vectors that we insert into the database globally.
4) IN_INSERT contains the incoming insertion requests.
5) STOP is the conditionally issued STOP request. This is placed last so that we only conditionally check it, as MPI performance is directly correlated with checking
   as few requests as possible. 

Whenever test() is called, we check all requests and then multiplex out which ones have actually been finished so that the caller can process them.     
**/
// clang-format on

class MPIRequest {
private:
  /**
     requests. This contains the current set of MPI requests that are handled by
  this class. This vector is divided into contiguous chunks, with each chunk
  representing a separate set of requests.
  **/
  std::vector<MPI_Request> requests;

  /**
     active_requests. This vector is used as scratch space for the progress
  tracking in this class.
  **/
  std::vector<int> active_requests;

  // This class has a particular internal ordering for events that has been
  // produced as the result of various optimisations. Briefly, we represent each
  // message type by a pair of unsigned integers in the "Event" struct. We use
  // these to identify how many requests are completed after a given iteration
  // and where in `active_requests` they live. These are organised in the
  // request_info array (see below).
  // Event is comprised of a `pos` field and a `size` field.
  // `size` is the number of completed requests in the most recent iteration of
  // `test` and `pos` is where the requests start in the `active_requests`
  // vector. This is used to allow us to discern which rank a particular request
  // corresponds to. N.B We benchmarked this layout against two separate arrays.
  // In practice, this appears to be more efficient (loading a request set
  // requires a single 64-bit load from a contiguous array, whereas two separate
  // arrays costs two loads. However, two separate arrays does lead to a lower
  // clearing cost, as only half as many bits need to be set).
  struct Event {
    unsigned pos{};
    unsigned size{};
  };

  // We index these events as follows.
  // N.B This ordering matters: if you change it, make sure you change
  // the test function too!
  enum class EventType : unsigned {
    BUCKET_OUT = 0,
    SIZES = 1,
    INSERTION_REQ = 2,
    BUCKET_IN = 3,
    STOP = 4,
    LAST = 5,
  };

  // This is the number of "fixed" event types. These are the event types that
  // rely on the number of ranks in the communicator, rather than any property
  // of those ranks.
  static constexpr unsigned number_of_request_types =
      static_cast<unsigned>(EventType::LAST);

  // This contains all of the events for a particular iteration.
  std::array<Event, number_of_request_types> request_info{};

  // We also need to keep track of where each fixed event type "starts" relative
  // to the requests buffer. We do this using a fixed size array that we set up
  // exactly once. This is never changed as the request layout is fixed at
  // creation.
  std::array<unsigned, number_of_request_types> fixed_starts{};

  // We provide the various setup steps as static functions to make it easier to
  // reason about what this class is doing.
  /**
     make_fixed_starts. This returns the "fixed_starts" array from a set of rank
  information.
     @param[in] nr_ranks: the number of ranks in the program.
     @param[in] batches: the number of batches in flight at once.
     @return an array that contains segment information for "requests".
  **/
  static std::array<unsigned, number_of_request_types>
  make_fixed_starts(const unsigned nr_ranks, const unsigned batches) noexcept;

  // This is just here to stop us from issuing extra memsets that we don't
  // really need.
  bool was_zeroed{};

public:
  /**
     ReqSpan. All completed requests produced by this class are represented as
     a ReqSpan, which is a pointer / size pair. The pointer points to a portion
  of memory containing `size` integers: data[i] is the rank whose request has
  been completed.
  **/
  struct ReqSpan {
    /**
       data. This is the pointer to the completed requests. This is never null:
       if size == 0, then this is points to the beginning of the active_requests
    vector.
    **/
    int *data;

    /**
       size. This is the number of completed requests in this request span. If
    this is zero then no requests were completed in this iteration.
    **/
    unsigned size;

    // These are all just standard iterators.
    int *begin() noexcept { return data; }
    int *end() noexcept { return data + size; }
    const int *begin() const noexcept { return data; }
    const int *end() const noexcept { return data + size; }
    int *cbegin() const noexcept { return data; }
    int *cend() const noexcept { return data + size; }

    int operator[](const unsigned pos) const noexcept { return data[pos]; }
  };

  explicit MPIRequest() noexcept;
  explicit MPIRequest(const unsigned nr_ranks,
                      const unsigned scale_factor) noexcept;

  constexpr static inline unsigned
  number_of_fixed_requests(const unsigned nr_ranks,
                           const unsigned batches) noexcept;
  constexpr static inline size_t
  number_of_bytes_needed(const unsigned nr_ranks,
                         const unsigned batches) noexcept;

  inline size_t number_of_requests() const noexcept;

  inline MPI_Request *size_requests(const unsigned batch = 0) noexcept;
  inline MPI_Request *bucketing_requests(const unsigned batch = 0) noexcept;
  inline MPI_Request *bucketing_replies(const unsigned batch = 0) noexcept;
  inline MPI_Request *insertion_requests(const unsigned batch = 0) noexcept;

  inline MPI_Request *stop_req() noexcept;

  inline unsigned nr_finished_insertions() const noexcept;
  inline unsigned nr_finished_stop_req() const noexcept;

  inline ReqSpan finished_insertions() noexcept;
  inline ReqSpan finished_size_reqs() noexcept;
  inline ReqSpan finished_bucketing_reqs() noexcept;
  inline ReqSpan finished_bucketing_reps() noexcept;
  inline ReqSpan finished_stop_requests() noexcept;

  inline void clear_reqs() noexcept;
  void test(const int rank, const int is_stop_inactive) noexcept;

private:
  inline void clear_all() noexcept;
  template <EventType index> inline MPI_Request *get_request() noexcept;
  template <EventType index> inline ReqSpan get_finished_span() noexcept;
  template <EventType index> inline unsigned get_finished_size() const noexcept;
  using IterType = decltype(active_requests)::iterator;

  template <EventType index>
  void adjust_span(IterType &start, const int outcount,
                   const int rank) noexcept;
};

#include "mpi_request.inl"

#endif
