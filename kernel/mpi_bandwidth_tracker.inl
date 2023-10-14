#ifndef INCLUDED_MPI_BANDWIDTH_TRACKER_HPP
#error Do not include mpi_bandwidth_tracker.inl without mpi_bandwidth_tracker.hpp
#endif

void MPIBandwidthTracker::add_count(const ContextChange type,
                                    const uint64_t bytes) noexcept {
  assert(type != ContextChange::LAST);
  const auto pos = static_cast<unsigned>(type);
  counts_[pos].data += bytes;
  ++counts_[pos].count;
}

void MPIBandwidthTracker::add_count_no_track(const ContextChange type,
                                             const uint64_t bytes) noexcept {
  assert(type != ContextChange::LAST);
  const auto pos = static_cast<unsigned>(type);
  counts_[pos].data += bytes;
}

void MPIBandwidthTracker::print_counts(std::ostream &os) noexcept {
  os << "type, bytes, count\n";
  for (unsigned i = 0; i < counts_.size(); i++) {
    os << static_cast<ContextChange>(i) << "," << counts_[i].data << ","
       << counts_[i].count << '\n';
  }
}

std::array<MPIBandwidthTracker::Event, MPIBandwidthTracker::size>
MPIBandwidthTracker::get_counts() noexcept {
  return counts_;
}

inline void MPIBandwidthTracker::clear_counts() noexcept {
  std::fill(counts_.begin(), counts_.end(), Event{});
}
