#ifndef INCLUDED_MPI_TIMER_HPP
#define INCLUDED_MPI_TIMER_HPP

#include "context_change.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>

// Forward declaration for API.
class TimerPoint;

/**
   MPITimer. This class is used to time various operations that are carried out
inside MPI.

   At a high-level, this class is used to time how long certain MPI operations
take from start to finish. In practice, this class is better used for MPI
operations that are "static" i.e. those operations that use blocking I/O, as
otherwise tracking time is a little bit difficult.

   This class works as follows: each event is tracked as a pair of a time and an
event count. This does preclude certain statistics, but this seems to be useful
in any case.

   The events themselves are registered via their event ID by the caller. We use
a simple RAII wrapper to actually clear up the event at the end, which marks the
timing.
**/
class MPITimer {
public:
  // This is the timer type that's being used. You can adjust this without
  // issue.
  using Tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
  // This is the resolution of all printed timings.
  using TimeResolution = std::chrono::milliseconds;
  // C++20 allows you to print the time with the suffix directly, but alas.
  constexpr static const char *const suffix = "ms";

  /**
     Time. This method returns a newly allocated TimerPoint object for a
  particular "event". This function does not throw.
     @param[in] event: the event kind that is being tracked. Must not be
  ContextChange::LAST.
     @return a new timer for that event.
  **/
  inline TimerPoint time(const ContextChange event) noexcept;

  /**
     add_time. This adds the time in `time` to the `event`.
     @param[in] event: the event to count. Must not be ContextChange::LAST.
     @param[in] time: the time to count.
  **/
  inline void add_time(const ContextChange event, const Tp time) noexcept;

  /**
     print_timings. This function just prints the timings stored in this class.
     @param[in] os: the output stream to use.
  **/
  inline void print_timings(std::ostream &os = std::cout) noexcept;

  inline std::array<uint64_t, 2>
  get_timing(const ContextChange event) const noexcept;

  inline uint64_t get_time(const ContextChange event) const noexcept;

  inline void reset_time() noexcept;

private:
  // Just the size of the array.
  static constexpr auto size = static_cast<unsigned>(ContextChange::LAST);

  struct Event {
    TimeResolution time;
    uint64_t count;
  };

  /**
     Timings. This array just contains the timings for this class.
  **/
  std::array<Event, size> timings;
};

struct TimerPoint {
  MPITimer::Tp time;
  ContextChange event;
  MPITimer *root;

  ~TimerPoint() { root->add_time(event, time); }
};

#include "mpi_timer.inl"

#endif
