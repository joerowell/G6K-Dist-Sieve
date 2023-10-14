#ifndef INCLUDED_MPI_BANDWIDTH_TRACKER_HPP
#define INCLUDED_MPI_BANDWIDTH_TRACKER_HPP

#include "context_change.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

/**
   MPIBandwidthTracker. This class is used to track the bandwidth of certain
operations that are carried out inside MPI.

   A large caveat: this class is a "best-case" estimate of how much data is
sent. This is because user-level tools are not adequate to track how much data
is being sent: there are many extra layers in networking and MPI that makes it
difficult to accurately gauge how much bandwidth is used. If you wish to
accurately measure how much data is sent, then we recommend this tool:

   https://github.com/Ivlyth/process-bandwidth

   Note: in order to avoid double-counting, this class should really only be
used to track outgoing data, not incoming data.
**/
class MPIBandwidthTracker {
public:
  /**
     add_count. This function adds `bytes` to the count for `type` and
  increments the number of sent `type` messages.
     @param[in] type: the type of message sent. Must not be ContextChange::LAST.
     @param[in] bytes: the number of bytes sent.
  **/
  inline void add_count(const ContextChange type,
                        const uint64_t bytes) noexcept;

  /**
     add_count_no_track. This function adds `bytes` to the count for `type`.
   Notably, this function does not increment the number of messages sent. This
   is typically useful for tracking headers.
     @param[in] type: the type of message sent. Must not be ContextChange::LAST.
     @param[in] bytes: the number of bytes sent.
   **/
  inline void add_count_no_track(const ContextChange type,
                                 const uint64_t bytes) noexcept;

  /**
     print_counts. This function just prints the counts_ stored in this class.
     @param[in] os: the output stream to use.
  **/
  inline void print_counts(std::ostream &os = std::cout) noexcept;

  /**
     Event. This struct counts the amount of data serialised and the number of
     times a particular message type was sent.
  **/
  struct Event {
    uint64_t data;
    uint64_t count;
  };

  // Just the size of the array.
  static constexpr auto size = static_cast<unsigned>(ContextChange::LAST);

  /**
     get_count. This function returns a copy of the counts on this particular
  node. This function does not throw.
     @return a copy of the counts_ array.
  **/
  inline std::array<Event, size> get_counts() noexcept;

  /**
     clear_counts. This function resets all of the counts to zero. This function
     does not throw.
   **/
  inline void clear_counts() noexcept;

private:
  /**
     counts_. This array just contains the events for this class.
  **/
  std::array<Event, size> counts_;
};

// Inline definitions live here.
#include "mpi_bandwidth_tracker.inl"

#endif
