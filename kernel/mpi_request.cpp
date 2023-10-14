#include "mpi_request.hpp"
// We'll only include these if we're doing MPI things
#include "context_change.hpp"
#ifndef DOCTEST_CONFIG_DISABLE
#include "doctest/extensions/doctest_mpi.h"
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <numeric>

std::array<unsigned, MPIRequest::number_of_request_types>
MPIRequest::make_fixed_starts(const unsigned nr_ranks,
                              const unsigned batches) noexcept {
  assert(nr_ranks > 1);
  const auto nr_bucket_out = batches;
  const auto nr_size_requests = batches;
  const auto nr_insertions = batches;
  const auto nr_bucket_in = batches;

  std::array<unsigned, MPIRequest::number_of_request_types> info;
  static_assert(unsigned(EventType::BUCKET_OUT) == 0);
  info[0] = 0;
  static_assert(unsigned(EventType::SIZES) == 1);
  info[1] = nr_bucket_out;
  static_assert(unsigned(EventType::INSERTION_REQ) == 2);
  info[2] = info[1] + nr_size_requests;
  static_assert(unsigned(EventType::BUCKET_IN) == 3);
  info[3] = info[2] + nr_insertions;
  static_assert(unsigned(EventType::STOP) == 4);
  info[4] = info[3] + nr_bucket_in;
  return info;
}

MPIRequest::MPIRequest() noexcept {}

MPIRequest::MPIRequest(const unsigned nr_ranks,
                       const unsigned scale_factor) noexcept
    : requests{std::vector<MPI_Request>(
          number_of_fixed_requests(nr_ranks, scale_factor), MPI_REQUEST_NULL)},
      active_requests{
          std::vector<int>(number_of_fixed_requests(nr_ranks, scale_factor))},
      request_info{}, fixed_starts{make_fixed_starts(nr_ranks, scale_factor)} {}

static inline int retrieve_offset(const int i, const int rank) noexcept {
  return i - (i >= rank);
}

static inline int retrieve_index(const int i, const int rank) noexcept {
  return i + (i >= rank);
}

template <MPIRequest::EventType index>
void MPIRequest::adjust_span(IterType &start, const int outcount,
                             const int rank) noexcept {

  constexpr auto pos = static_cast<unsigned>(index);
  static_assert(index != EventType::STOP);

  // Work out how many requests in this span actually were for `index`.
  const auto bound = fixed_starts[pos + 1];
  request_info[pos].size = static_cast<unsigned>(
      std::lower_bound(start, active_requests.begin() + outcount, bound) -
      start);

  const auto upper = start + request_info[pos].size;
  request_info[pos].pos = start - active_requests.cbegin();

  // This loop just subtracts the starting position: the index type doesn't
  // really matter.

  while (start != upper) {
    *start = *start - fixed_starts[pos];
    ++start;
  }
}

void MPIRequest::test(const int rank, const int is_stop_inactive) noexcept {

  assert(active_requests.size() == requests.size());
  int outcount;
  MPI_Testsome(static_cast<int>(requests.size()) - is_stop_inactive,
               requests.data(), &outcount, active_requests.data(),
               MPI_STATUSES_IGNORE);

  // If there's no active request MPI_Testsome can return MPI_UNDEFINED, which
  // we have to guard against.
  if (outcount == 0 || outcount == MPI_UNDEFINED) {
    clear_all();
    return;
  }

  // The array hasn't been zeroed, so mark is as such.
  was_zeroed = false;

  // We now need to filter through those that were finished. We do this by
  // sorting the first `outcount` values in active_requests.
  assert(outcount < static_cast<int>(active_requests.size()));
  std::sort(active_requests.begin(), active_requests.begin() + outcount);

  // We now search over the active_requests and divide them up into
  // reasonable blocks.
  // Intuitively, this code works as follows: if any entry in active_requests is
  // less than batches, then it must have corresponded to a BUCKET_IN request
  // that finished. Similarly, if any entry in active_requests is less than 2 *
  // batches, then it must have been a size request, and so on. Because we
  // encode this information into fixed_starts (implicitly earlier) we just need
  // to adjust the relevant spans to represent which request was done, and then
  // all is good.

  auto curr = active_requests.begin();
  adjust_span<EventType::BUCKET_OUT>(curr, outcount, rank);
  adjust_span<EventType::SIZES>(curr, outcount, rank);
  adjust_span<EventType::INSERTION_REQ>(curr, outcount, rank);
  adjust_span<EventType::BUCKET_IN>(curr, outcount, rank);

  // Check the last one too. This requires some care: it's technically possible
  // that active_requests[outcount-1] is set from a previous barrier that
  // succeeded. To circumvent this, we only mark as being finished if the
  // barrier was issued, if the iterator points to the right value, and if the
  // iterator is valid. This prevents a very rare bug.
  request_info[unsigned(EventType::STOP)].size =
      !is_stop_inactive &&
      static_cast<unsigned>(*curr) == requests.size() - 1 &&
      curr != active_requests.begin() + outcount;
}
