#ifndef INCLUDED_BGJ1_BUCKETING_INTERFACE_HPP
#error Do not include bgj1_bucketing_interface.inl without bgj1_bucketing_interface.hpp
#endif

inline void
bgj1_bucketing_interface::inc_sieved_buckets(const unsigned batch) noexcept {
  ops_[batch].fetch_add(1, std::memory_order_relaxed);
}

inline void
bgj1_bucketing_interface::inc_bucketed_count(const unsigned batch) noexcept {
  // N.B This has to be a release, as it also flushes all threaded
  // writes. In some settings (i.e. if there's been a batched write
  // of some kind) this could probably actually be relaxed, but
  // this is safer.
  ops_[batch].fetch_add(1, std::memory_order_release);
}

inline Communicator &
bgj1_bucketing_interface::get_comm(const unsigned batch) noexcept {
  return comms_[batch];
}

inline void bgj1_bucketing_interface::start_incoming_centers(
    const unsigned batch) noexcept {
  ops_[batch].store(0, std::memory_order_relaxed);
  comms_[batch].start_bucketing();
  ++counts_[batch];
}

inline void
bgj1_bucketing_interface::start_incoming_size(const unsigned batch) noexcept {
  comms_[batch].start_size();
  ops_[batch].store(0, std::memory_order_relaxed);
}

inline std::vector<char> &
bgj1_bucketing_interface::get_memory_for(const unsigned batch) noexcept {
  return sync_headers_[batch];
}

inline uint64_t bgj1_bucketing_interface::bucket_position(
    const unsigned rank, const unsigned batch, const unsigned width) noexcept {
  return bucket_positions_[(width * rank) + batch];
}

inline std::vector<int> &
bgj1_bucketing_interface::get_sizes(const unsigned batch) noexcept {
  return sizes_[batch];
}

inline bool bgj1_bucketing_interface::has_finished_sieving_ours(
    const unsigned batch, const unsigned buckets) noexcept {
  // Here we don't really need certain writes to be finished, so we can just
  // load the counter.
  return comms_[batch].is_sieving() &&
         buckets == ops_[batch].load(std::memory_order_relaxed);
}

inline void
bgj1_bucketing_interface::start_sieving_bucket(const unsigned batch) noexcept {
  // This can also be relaxed because this is just to signal to other threads
  // that everything is fine (the underlying MPI implementation should make the
  // writes etc synchronous).
  ops_[batch].store(0, std::memory_order_relaxed);
  comms_[batch].start_sieving();
}

inline void bgj1_bucketing_interface::mark_sieving_as_finished(
    const unsigned batch) noexcept {
  ops_[batch].store(0, std::memory_order_relaxed);
}

inline void bgj1_bucketing_interface::reset_ops() noexcept {

  for (auto &op : ops_) {
    op.store(0, std::memory_order_relaxed);
  }

  std::atomic_thread_fence(std::memory_order_seq_cst);
}

inline std::vector<uint8_t> &bgj1_bucketing_interface::states() noexcept {
  return states_;
}

inline std::vector<uint8_t> &bgj1_bucketing_interface::counts() noexcept {
  return counts_;
}

inline bool
bgj1_bucketing_interface::can_issue_more(const unsigned batch) const noexcept {
  return states_[batch] != counts_[batch];
}

inline void bgj1_bucketing_interface::reset_counts() noexcept {
  std::fill(counts_.begin(), counts_.end(), 0);
  std::fill(states_.begin(), states_.end(), 0);
}

inline void bgj1_bucketing_interface::reset_comms() noexcept {
  for (auto &comm : comms_) {
    comm.reset();
  }
}

inline void bgj1_bucketing_interface::bucketing_request_finished(
    const unsigned batch) noexcept {
  comms_[batch].start_sieving();
  ops_[batch].store(0, std::memory_order_relaxed);
}

inline bool bgj1_bucketing_interface::has_finished_bucketing(
    const unsigned batch, const unsigned nr_ranks) noexcept {
  // N.B This can be relaxed because we will synchronise with the release
  // barrier before we serialise.
  return comms_[batch].is_bucketing() &&
         ops_[batch].load(std::memory_order_relaxed) == nr_ranks;
}

inline void bgj1_bucketing_interface::start_incoming_buckets(
    const unsigned batch) noexcept {

  comms_[batch].start_incoming_buckets();
  ops_[batch].store(0, std::memory_order_relaxed);
}

inline bool bgj1_bucketing_interface::are_all_done() const noexcept {
  const auto size = comms_.size();
  for (unsigned i = 0; i < size; i++) {
    if (!comms_[i].is_reset() || states_[i] != counts_[i]) {
      return false;
    }
  }
  return true;
}

inline bool
bgj1_bucketing_interface::is_sizing(const unsigned batch) const noexcept {
  return comms_[batch].is_sizing();
}

inline unsigned bgj1_bucketing_interface::get_next_wait_head(
    const unsigned start) const noexcept {

  // We can only use the next one if it is sieving and the counts don't match,
  // if the communicator is receiving buckets or sizes and the counts do match,
  // or if we're receiving centers (mismatches are impossible here).

  const auto check_func = [&](const unsigned pos) {
    const auto are_same = states_[pos] == counts_[pos];
    return (!are_same && (comms_[pos].is_sieving() ||
                          comms_[pos].is_receiving_insertion_sizes() ||
                          comms_[pos].is_receiving_insertions() ||
                          comms_[pos].is_inserting())) ||
           (are_same &&
            (comms_[pos].is_bucketing() || comms_[pos].is_sizing())) ||
           comms_[pos].has_incoming_centers();
  };

  // Always start at the next one.
  auto pos = (start == comms_.size() - 1) ? 0 : start + 1;
  for (unsigned i = 0; i < comms_.size(); i++) {
    if (check_func(pos)) {
      return pos;
    }
    pos = (pos == comms_.size() - 1) ? 0 : pos + 1;
  }

  // We must be done.
  return comms_.size();
}

inline void bgj1_bucketing_interface::free_comms() noexcept {
  for (auto &comm_ : comms_) {
    comm_.free_comm();
  }
}

inline uint64_t bgj1_bucketing_interface::bytes_used() const noexcept {
  return comms_.capacity() * sizeof(decltype(comms_)::value_type) +
         ops_.capacity() * sizeof(decltype(ops_)::value_type) +
         states_.capacity() * sizeof(decltype(states_)::value_type) +
         counts_.capacity() * sizeof(decltype(counts_)::value_type) +
         sync_headers_.capacity() *
             sizeof(decltype(sync_headers_)::value_type) +
         std::accumulate(sync_headers_.cbegin(), sync_headers_.cend(),
                         uint64_t(0),
                         [](const uint64_t lhs, const std::vector<char> &rhs) {
                           return lhs + rhs.capacity() * sizeof(char);
                         }) +
         sizes_.capacity() * sizeof(decltype(sizes_)::value_type) +
         std::accumulate(sizes_.cbegin(), sizes_.cend(), uint64_t(0),
                         [](const uint64_t lhs, const std::vector<int> &rhs) {
                           return lhs + rhs.capacity() * sizeof(int);
                         }) +
         bucket_positions_.capacity() *
             sizeof(decltype(bucket_positions_)::value_type);
}

inline std::vector<Communicator> &bgj1_bucketing_interface::comms() noexcept {
  return comms_;
}

inline unsigned
bgj1_bucketing_interface::get_next_incoming_centres() const noexcept {
  const auto iter = std::find_if(
      comms_.cbegin(), comms_.cend(),
      [](const Communicator &comm) { return comm.is_receiving_buckets(); });
  return std::distance(comms_.cbegin(), iter);
}

inline unsigned bgj1_bucketing_interface::get_next_sizes() const noexcept {
  const auto iter =
      std::find_if(comms_.cbegin(), comms_.cend(),
                   [](const Communicator &comm) { return comm.is_sizing(); });
  return std::distance(comms_.cbegin(), iter);
}

inline bool
bgj1_bucketing_interface::is_sieving(const unsigned index) const noexcept {
  return comms_[index].is_sieving();
}

inline bool bgj1_bucketing_interface::is_receiving_buckets(
    const unsigned index) const noexcept {
  return comms_[index].is_receiving_buckets();
}

inline void bgj1_bucketing_interface::mark_insertion_as_done(
    const unsigned index) noexcept {
  ops_[index].store(1, std::memory_order_release);
}

inline void bgj1_bucketing_interface::mark_batch_as_finished(
    const unsigned index) noexcept {
  // No need to store here: it's done in start_incoming_centers.
  comms_[index].reset();
}

inline bool bgj1_bucketing_interface::is_receiving_insertion_sizes(
    const unsigned index) const noexcept {
  return comms_[index].is_receiving_insertion_sizes();
}

inline void
bgj1_bucketing_interface::start_insertions(const unsigned batch) noexcept {
  comms_[batch].start_insertions();
}

inline bool bgj1_bucketing_interface::has_finished_insertion(
    const unsigned index) const noexcept {
  return comms_[index].is_inserting() &&
         ops_[index].load(std::memory_order_relaxed) == 1;
}

inline void bgj1_bucketing_interface::start_incoming_insertion_size(
    const unsigned batch) noexcept {
  ops_[batch].store(0, std::memory_order_relaxed);
  comms_[batch].start_incoming_insertion_size();
}

inline void bgj1_bucketing_interface::start_incoming_insertion(
    const unsigned batch) noexcept {
  comms_[batch].start_insertion_exchange();
}

inline bool
bgj1_bucketing_interface::is_finished(const unsigned batch) const noexcept {
  return comms_[batch].is_reset() && states_[batch] == counts_[batch];
}
