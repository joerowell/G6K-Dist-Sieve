#ifndef INCLUDED_MPI_OBJ_HPP
#error Do not include MPI_obj.inl without MPI_obj.hpp
#endif

inline constexpr bool MPIObj::is_distributed_sieving_enabled() noexcept {
#ifdef G6K_MPI
  return true;
#else
  return false;
#endif
}

inline G6K_MPI_CONDITIONAL_CONSTEXPR bool MPIObj::is_root() const noexcept {
#ifdef G6K_MPI
  return rank == 0;
#else
  return false;
#endif
}

inline G6K_MPI_CONDITIONAL_CONSTEXPR int MPIObj::get_rank() const noexcept {
#ifdef G6K_MPI
  return rank;
#else
  return 0;
#endif
}

inline int MPIObj::number_of_ranks() const noexcept {
#ifdef G6K_MPI
  assert([this]() {
    int ranks;
    MPI_Comm_size(comm, &ranks);
    return static_cast<unsigned>(ranks);
  }() == memory_per_rank.size());
  return memory_per_rank.size();
#else
  return 0;
#endif
}

inline G6K_MPI_CONDITIONAL_CONSTEXPR DistSieverType
MPIObj::get_topology() const noexcept {
#ifdef G6K_MPI
  return topology->get_type();
#else
  return DistSieverType::Null;
#endif
}

inline G6K_MPI_CONDITIONAL_CONSTEXPR bool MPIObj::is_active() const noexcept {
#ifdef G6K_MPI
  return active;
#else
  return false;
#endif
}

inline G6K_MPI_CONDITIONAL_CONSTEXPR bool
MPIObj::should_sieve(const unsigned n,
                     const unsigned threshold) const noexcept {
#ifdef G6K_MPI
  return n >= threshold;
#else
  (void)n;
  (void)threshold;
  return false;
#endif
}

inline G6K_MPI_CONDITIONAL_CONSTEXPR bool
MPIObj::is_in_context_change() const noexcept {
#ifdef G6K_MPI
  return state == MPIObj::State::CONTEXT_CHANGE;
#else
  return false;
#endif
}

inline void MPIObj::out_of_context_change() noexcept {
#ifdef G6K_MPI
  state = MPIObj::State::DEFAULT;
#endif
}

inline void MPIObj::in_context_change() noexcept {
#ifdef G6K_MPI
  state = MPIObj::State::CONTEXT_CHANGE;
#endif
}

inline G6K_MPI_CONDITIONAL_CONSTEXPR bool MPIObj::is_sieving() const noexcept {
#ifdef G6K_MPI
  return state == MPIObj::State::SIEVING;
#else
  return false;
#endif
}

inline void MPIObj::in_sieving() noexcept {
#ifdef G6K_MPI
  state = MPIObj::State::SIEVING;
#endif
}

inline void MPIObj::out_of_sieving() noexcept {
#ifdef G6K_MPI
  state = MPIObj::State::DEFAULT;
#endif
}

inline G6K_MPI_CONDITIONAL_CONSTEXPR uint64_t
MPIObj::get_total_buckets() const noexcept {
#ifdef G6K_MPI
  return total_bucket_count;
#else
  return 0;
#endif
}

inline unsigned MPIObj::retrieve_index(const unsigned i,
                                       const unsigned my_rank) noexcept {
#ifdef G6K_MPI
  const auto res = i + (i >= my_rank);
  assert(res != my_rank);
  return res;
#else
  (void)i;
  (void)my_rank;
  return 0;
#endif
}

inline unsigned MPIObj::retrieve_offset(const unsigned i,
                                        const unsigned my_rank) noexcept {
#ifdef G6K_MPI
  assert(i != my_rank);
  return i - (i > my_rank);
#else
  (void)i;
  (void)my_rank;
  return 0;
#endif
}

inline std::array<uint64_t, 2>
MPIObj::get_size_and_start(const unsigned o_rank,
                           const unsigned batch) noexcept {
#ifdef G6K_MPI
  return {memory_per_rank[o_rank], get_bucket_position(o_rank, batch)};
#else
  (void)o_rank;
  return {};
#endif
}

inline void MPIObj::inc_sieved_buckets(const unsigned index) noexcept {
#ifdef G6K_MPI
  bgj1_buckets.inc_sieved_buckets(index);
#endif
}

inline bool MPIObj::has_finished_sieving_ours(const unsigned index) noexcept {
#ifdef G6K_MPI
  return bgj1_buckets.has_finished_sieving_ours(index, get_bucket_size(rank));
#else
  return true;
#endif
}

inline void MPIObj::issue_stop_barrier() noexcept {
#ifdef G6K_MPI
  is_barrier_empty = false;
  MPI_Ibarrier(stop_comm, requests.stop_req());
#endif
}

inline bool MPIObj::is_done() noexcept {
#ifdef G6K_MPI
  return issued_stop && stopped;
#endif
}

#if defined G6K_MPI
inline std::vector<uint64_t> MPIObj::get_memory_per_rank() noexcept {
  return memory_per_rank;
}

inline uint64_t MPIObj::get_buckets_per_rank() noexcept {
  return buckets_per_rank;
}
#endif

#if defined G6K_MPI
inline uint64_t MPIObj::get_bucket_position(const unsigned o_rank,
                                            const unsigned batch) noexcept {
  return bgj1_buckets.bucket_position(o_rank, batch, bucket_batches);
}
#endif

inline size_t MPIObj::get_nr_buckets() const noexcept {
#ifdef G6K_MPI
  return buckets_per_rank;
#else
  __builtin_unreachable();
#endif
}

inline std::vector<ZT> &
MPIObj::get_insertion_candidates(const unsigned batch) noexcept {
#ifdef G6K_MPI
  return insertion_vector_bufs[batch].incoming_buffer;
#else
  __builtin_unreachable();
#endif
}

inline void MPIObj::mark_insertion_as_done(const unsigned batch) noexcept {
#ifdef G6K_MPI
  bgj1_buckets.mark_insertion_as_done(batch);
#else
  __builtin_unreachable();
#endif
}

inline void MPIObj::set_sat_drop(const unsigned drop) noexcept {
#ifdef G6K_MPI
  sat_drop.fetch_add(drop, std::memory_order_relaxed);
#endif
}

inline bool MPIObj::has_stopped() const noexcept {
#ifdef G6K_MPI
  return issued_stop;
#else
  return false;
#endif
}

inline bool MPIObj::can_issue_barrier() const noexcept {
#ifdef G6K_MPI
  return is_barrier_empty && !issued_stop;
#endif
}

inline bool MPIObj::has_barrier_finished() const noexcept {
#ifdef G6K_MPI
  return stopped;
#else
  return false;
#endif
}

inline bool MPIObj::is_initial_barrier_completed() const noexcept {
#ifdef G6K_MPI
  return stopped && !issued_stop;
#else
  return false;
#endif
}

inline void MPIObj::start_sieving_bucket(const unsigned index) noexcept {
#ifdef G6K_MPI
  bgj1_buckets.start_sieving_bucket(index);
#endif
}

inline bool MPIObj::can_issue_more(const unsigned batch) const noexcept {
#ifdef G6K_MPI
  return !issued_stop || bgj1_buckets.can_issue_more(batch);
#endif
}

inline MPIRequest::ReqSpan MPIObj::finished_buckets() noexcept {
#ifdef G6K_MPI
  return requests.finished_bucketing_reps();
#endif
}

inline void MPIObj::bucketing_request_finished(const unsigned batch) noexcept {
#ifdef G6K_MPI
  // Free the memory for the outgoing batch.
  auto &buf = scratch_space[scratch_lookup[batch]];
  buf.outgoing_scratch.clear();
  buf.outgoing_scratch.shrink_to_fit();

  bgj1_buckets.bucketing_request_finished(batch);
#endif
}

inline void MPIObj::mark_sieving_as_finished(const unsigned batch) noexcept {
#ifdef G6K_MPI
  bgj1_buckets.mark_sieving_as_finished(batch);
#endif
}

inline MPIRequest::ReqSpan MPIObj::finished_sieving() noexcept {
#ifdef G6K_MPI
  unsigned size = 0;
  for (unsigned i = 0; i < bucket_batches; i++) {
    if (has_finished_sieving_ours(i)) {
      finished_batches[size] = i;
      ++size;
    }
  }
  return {finished_batches.data(), size};
#endif
  __builtin_unreachable();
}

inline MPIRequest::ReqSpan MPIObj::finished_insertion_sizes() noexcept {
#ifdef G6K_MPI
  unsigned size = 0;
  for (unsigned i = 0; i < bucket_batches; i++) {
    if (finished_sizes[i] && bgj1_buckets.is_receiving_insertion_sizes(i)) {
      finished_batches[size] = i;
      finished_sizes[i] = false;
      ++size;
    }
  }
  return {finished_batches.data(), size};
#else
  __builtin_unreachable();
#endif
}

inline MPIRequest::ReqSpan MPIObj::finished_incoming_insertions() noexcept {
#ifdef G6K_MPI
  return requests.finished_insertions();
#endif
}

inline MPIRequest::ReqSpan MPIObj::finished_processed_buckets() noexcept {
#ifdef G6K_MPI
  unsigned size = 0;

  for (unsigned i = 0; i < bucket_batches; i++) {
    if (bgj1_buckets.has_finished_bucketing(i, nr_real_buckets)) {
      finished_batches[size] = i;
      ++size;
    }
  }
  return {finished_batches.data(), size};
#endif
  __builtin_unreachable();
}

inline bool MPIObj::can_issue_stop_barrier() const noexcept {
#ifdef G6K_MPI
  return issued_stop && is_barrier_empty && bgj1_buckets.are_all_done();
#else
  __builtin_unreachable();
#endif
}

inline MPIRequest::ReqSpan MPIObj::finished_syncs() noexcept {
#ifdef G6K_MPI
  return requests.finished_bucketing_reqs();
#else
  (void)o_rank;
  __builtin_unreachable();
#endif
}

inline void MPIObj::inc_bucketed_count(const unsigned batch) noexcept {
#ifdef G6K_MPI
  bgj1_buckets.inc_bucketed_count(batch);
#else
  (void)o_rank;
  (void)batch;
#endif
}

inline void MPIObj::abort(const int error) noexcept {
#ifdef G6K_MPI
  MPI_Abort(comm, error);
#endif
}

inline bool MPIObj::owns_uid(const UidType uid) const noexcept {
#ifdef G6K_MPI
  return slot_map[UidHashTable::get_slot(uid)] == rank;
#else
  return true;
#endif
}

inline MPIObj::ThreadEntry &
MPIObj::get_thread_entry(const unsigned id) noexcept {
#ifdef G6K_MPI
  return thread_entries[id];
#else
  assert(false);
  __builtin_unreachable();
#endif
}

inline MPIObj::BdglThreadEntry &
MPIObj::get_thread_entry_bdgl(const unsigned id) noexcept {
#ifdef G6K_MPI
  return bdgl_thread_entries[id];
#else
  assert(false);
  __builtin_unreachable();
#endif
}

inline uint64_t MPIObj::get_bucket_size(const unsigned o_rank) noexcept {
#ifdef G6K_MPI
  return memory_per_rank[o_rank];
#else
  __builtin_unreachable();
#endif
}

inline unsigned MPIObj::number_of_free_scratch_space() const noexcept {
#ifdef G6K_MPI
  return free_scratch.load(std::memory_order_relaxed);
#endif
}

inline void MPIObj::decrease_free_scratch(const unsigned val) noexcept {
#ifdef G6K_MPI
  free_scratch.fetch_sub(val, std::memory_order_relaxed);
#endif
}

inline unsigned MPIObj::get_batch_index(const unsigned index) noexcept {
#ifdef G6K_MPI
  return scratch_lookup[index];
#endif
}

inline void MPIObj::dec_batch_use(const unsigned index) noexcept {
#ifdef G6K_MPI
  if (scratch_used[index].fetch_sub(1, std::memory_order_acq_rel) == 1) {
    free_scratch.fetch_add(1, std::memory_order_release);
    scratch_space[index].incoming_scratch.clear();
    scratch_space[index].incoming_scratch.shrink_to_fit();
  }
#endif
}

inline MPIRequest::ReqSpan MPIObj::finished_insertions() noexcept {
#ifdef G6K_MPI
  unsigned size = 0;
  for (unsigned i = 0; i < bucket_batches; i++) {
    if (bgj1_buckets.has_finished_insertion(i)) {
      finished_batches[size] = i;
      ++size;
    }
  }
  return {finished_batches.data(), size};
#endif
  __builtin_unreachable();
}

inline void MPIObj::mark_batch_as_finished(const unsigned batch) noexcept {
#ifdef G6K_MPI
  insertion_vector_bufs[batch].incoming_buffer.clear();
  insertion_vector_bufs[batch].incoming_buffer.shrink_to_fit();
  bgj1_buckets.mark_batch_as_finished(batch);
#endif
}

inline void MPIObj::start_insertions(const unsigned batch) noexcept {
#ifdef G6K_MPI
  // Free the outgoing storage.
  insertion_vector_bufs[batch].outgoing_buffer.clear();
  insertion_vector_bufs[batch].outgoing_buffer.shrink_to_fit();
  bgj1_buckets.start_insertions(batch);
#endif
}

#if defined(MPI_TIME) && defined(G6K_MPI)
inline MPIObj::RecompTimer MPIObj::time_recomp() {
  return RecompTimer{std::chrono::steady_clock::now(), this};
}
#elif defined(G6K_MPI)
inline void MPIObj::time_recomp() {}
#endif

inline size_t MPIObj::get_number_of_bdgl_buckets() const noexcept {
#ifdef G6K_MPI
  return bdgl_bucket_map[rank].size();
#else
  __builtin_unreachable();
#endif
}

inline bool MPIObj::has_bdgl_finished_sieving() const noexcept {
#ifdef G6K_MPI
  return stopped;
#else
  return false;
#endif
}

inline void MPIObj::bdgl_pass_stop() noexcept {
#ifdef G6K_MPI
  stopped = false;
  is_barrier_empty = true;
  bdgl_nr_completed = 0;
  bdgl_nr_outstanding = 0;
  wait_head = 0;
  MPI_Type_free(&entry_type);
#endif
}

inline std::vector<bucket_pair> &
MPIObj::get_bucket_pairs(const unsigned batch) noexcept {
#ifdef G6K_MPI
  return bucket_pairs[batch];
#endif
}

inline bool MPIObj::bdgl_has_finished_contributing() const noexcept {
#ifdef G6K_MPI
  return bdgl_nr_completed == unsigned(number_of_ranks());
#endif
}

inline bool MPIObj::bdgl_has_finished() const noexcept {
#ifdef G6K_MPI
  return is_barrier_empty &&
         nr_remaining_buckets.load(std::memory_order_relaxed) == 0 &&
         bdgl_nr_completed == unsigned(number_of_ranks()) &&
         bdgl_nr_outstanding == 0;
#endif
}

inline size_t MPIObj::get_bdgl_batch_size(const unsigned index) const noexcept {
#ifdef G6K_MPI
  return bdgl_batch_info[index].size;
#else
  __builtin_unreachable();
#endif
}

inline unsigned MPIObj::get_bdgl_bucket(const unsigned index,
                                        const unsigned offset) const noexcept {
#ifdef G6K_MPI
  const auto start_elem = bdgl_batch_info[index].pos;
  return bdgl_bucket_map[rank][start_elem + offset];
#else
  __builtin_unreachable();
#endif
}

inline void MPIObj::bdgl_mark_bucket_as_processed() noexcept {
#ifdef G6K_MPI
  nr_remaining_buckets.fetch_sub(1, std::memory_order_relaxed);
#endif
}

inline void MPIObj::mark_bdgl_bucket_as_finished() noexcept {
#ifdef G6K_MPI
  nr_remaining_buckets.fetch_sub(1, std::memory_order_relaxed);
#endif
}

inline void MPIObj::bdgl_clear_thread_entry(const unsigned id) noexcept {
#ifdef G6K_MPI
  bdgl_thread_entries[id].t_queue.clear();
  bdgl_thread_entries[id].t_queue.shrink_to_fit();
#endif
}

inline int64_t MPIObj::get_bdgl_insert_pos(const unsigned id) noexcept {
#ifdef G6K_MPI
  return bdgl_insert_pos[id].load(std::memory_order_relaxed);
#endif
}

inline void MPIObj::bdgl_update_insert_pos(const unsigned id,
                                           const int64_t pos) noexcept {
#ifdef G6K_MPI
  return bdgl_insert_pos[id].store(pos, std::memory_order_relaxed);
#endif
}

inline void
MPIObj::bdgl_decrement_outstanding_batches(const unsigned size) noexcept {
#ifdef G6K_MPI
  bdgl_nr_outstanding -= size;
#endif
}

inline void MPIObj::bdgl_dec_bucket_use(const unsigned index) noexcept {
#ifdef G6K_MPI
  bdgl_bucket_active[index].fetch_sub(1, std::memory_order_relaxed);
#endif
}
