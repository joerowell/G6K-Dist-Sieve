#ifndef INCLUDED_MPI_REQUEST_HPP
#error Do not include mpi_request.inl without mpi_request.hpp
#endif

template <MPIRequest::EventType index>
inline unsigned MPIRequest::get_finished_size() const noexcept {
  return request_info[static_cast<unsigned>(index)].size;
}

inline unsigned MPIRequest::nr_finished_insertions() const noexcept {
  return get_finished_size<EventType::INSERTION_REQ>();
}

inline unsigned MPIRequest::nr_finished_stop_req() const noexcept {
  return get_finished_size<EventType::STOP>();
}

template <MPIRequest::EventType index>
inline MPI_Request *MPIRequest::get_request() noexcept {
  return &requests[fixed_starts[static_cast<unsigned>(index)]];
}

inline MPI_Request *
MPIRequest::insertion_requests(const unsigned batch) noexcept {
  return get_request<EventType::INSERTION_REQ>() + batch;
}

inline MPI_Request *MPIRequest::size_requests(const unsigned batch) noexcept {
  return get_request<EventType::SIZES>() + batch;
}

inline MPI_Request *
MPIRequest::bucketing_requests(const unsigned batch) noexcept {
  return get_request<EventType::BUCKET_OUT>() + batch;
}

inline MPI_Request *
MPIRequest::bucketing_replies(const unsigned batch) noexcept {
  return get_request<EventType::BUCKET_IN>() + batch;
}

inline MPI_Request *MPIRequest::stop_req() noexcept {
  return get_request<EventType::STOP>();
}

template <MPIRequest::EventType index>
inline MPIRequest::ReqSpan MPIRequest::get_finished_span() noexcept {
  const auto req_info = request_info[static_cast<unsigned>(index)];
  return {&active_requests[req_info.pos], req_info.size};
}

inline MPIRequest::ReqSpan MPIRequest::finished_insertions() noexcept {
  return get_finished_span<EventType::INSERTION_REQ>();
}

inline MPIRequest::ReqSpan MPIRequest::finished_size_reqs() noexcept {
  return get_finished_span<EventType::SIZES>();
}

inline MPIRequest::ReqSpan MPIRequest::finished_bucketing_reqs() noexcept {
  return get_finished_span<EventType::BUCKET_OUT>();
}

inline MPIRequest::ReqSpan MPIRequest::finished_bucketing_reps() noexcept {
  return get_finished_span<EventType::BUCKET_IN>();
}

inline void MPIRequest::clear_all() noexcept {
  if (was_zeroed) {
    return;
  }

  // There's not much difference between memset and std::fill here,
  // but memset will cause the compiler to moan.
  std::fill(request_info.begin(), request_info.end(), Event{});
  was_zeroed = true;
}

inline void MPIRequest::clear_reqs() noexcept {
  // This should be handled already by the caller, so this is really
  // just a failsafe.
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

constexpr inline unsigned
MPIRequest::number_of_fixed_requests(const unsigned nr_ranks,
                                     const unsigned batches) noexcept {
  assert(nr_ranks > 0);

  // Number of BUCKET_OUT requests.
  const auto nr_bucket_out = batches;

  // Number of size requests.
  const auto nr_size_requests = batches;

  // Number of outgoing and incoming insertions.
  const auto nr_insertions = batches;

  // Number of BUCKET_IN requests.
  const auto nr_bucket_in = batches;

  // Number of stop barriers.
  constexpr auto nr_stop_barriers = 1;

  return nr_bucket_out + nr_size_requests + nr_insertions + nr_bucket_in +
         nr_stop_barriers;
}

constexpr inline size_t
MPIRequest::number_of_bytes_needed(const unsigned nr_ranks,
                                   const unsigned batches) noexcept {
  const auto nr_requests = number_of_fixed_requests(nr_ranks, batches);
  return nr_requests * (sizeof(int) + sizeof(MPI_Request));
}
