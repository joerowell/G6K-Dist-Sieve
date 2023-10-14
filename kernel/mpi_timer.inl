#ifndef INCLUDED_MPI_TIMER_HPP
#error Do not include mpi_timer.inl without mpi_timer.hpp
#endif

inline TimerPoint MPITimer::time(const ContextChange event) noexcept {
  assert(event != ContextChange::LAST);
  return TimerPoint{std::chrono::high_resolution_clock::now(), event, this};
}

inline void MPITimer::add_time(const ContextChange event,
                               const Tp time) noexcept {
  assert(event != ContextChange::LAST);
  timings[static_cast<unsigned>(event)].time +=
      std::chrono::duration_cast<TimeResolution>(
          std::chrono::high_resolution_clock::now() - time);
  ++timings[static_cast<unsigned>(event)].count;
}

inline void MPITimer::print_timings(std::ostream &os) noexcept {

  for (unsigned i = 0; i < timings.size(); i++) {
    os << static_cast<ContextChange>(i) << " (count, timings, mean): ("
       << timings[i].count << "," << timings[i].time.count() << suffix << ","
       << (double)timings[i].time.count() / timings[i].count << ")";
  }
}

inline std::array<uint64_t, 2>
MPITimer::get_timing(const ContextChange state) const noexcept {
  assert(state != ContextChange::LAST);
  const auto pos = static_cast<unsigned>(state);
  return {uint64_t(timings[pos].time.count()), timings[pos].count};
}

inline uint64_t MPITimer::get_time(const ContextChange state) const noexcept {
  assert(state != ContextChange::LAST);
  return timings[static_cast<unsigned>(state)].time.count();
}

inline void MPITimer::reset_time() noexcept {
  for (auto &v : timings) {
    v.time = {};
    v.count = 0;
  }
}
