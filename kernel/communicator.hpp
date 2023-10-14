#ifndef INCLUDED_COMMUNICATOR_HPP
#define INCLUDED_COMMUNICATOR_HPP

#include "comm_state.hpp"
#include "mpi.h"
#include <cassert>

/**
   Communicator
**/

class Communicator {
public:
  inline Communicator() {}

  inline MPI_Comm &get_comm() noexcept;

  inline void duplicate_comm(MPI_Comm comm) noexcept;

  inline void start_incoming_centers() noexcept;
  inline bool has_incoming_centers() const noexcept;

  inline void start_bucketing() noexcept;
  inline bool is_bucketing() const noexcept;

  inline void start_size() noexcept;
  inline bool is_sizing() const noexcept;

  inline void mark_size_finished() noexcept;
  inline bool has_finished_size() const noexcept;

  inline void start_outgoing_centers() noexcept;
  inline bool is_sending_centers() const noexcept;

  inline void start_incoming_buckets() noexcept;
  inline bool is_receiving_buckets() const noexcept;

  inline void start_outgoing_buckets() noexcept;
  inline bool is_sending_buckets() const noexcept;

  inline void start_sieving() noexcept;
  inline bool is_sieving() const noexcept;

  inline void start_incoming_insertion_size() noexcept;
  inline bool is_receiving_insertion_sizes() const noexcept;

  inline void start_insertion_exchange() noexcept;
  inline bool is_receiving_insertions() const noexcept;

  inline void start_insertions() noexcept;
  inline bool is_inserting() const noexcept;

  inline void reset() noexcept;
  inline bool is_reset() const noexcept;

  inline CommState get_state() const noexcept;

  inline void free_comm() noexcept;

private:
  MPI_Comm comm{MPI_COMM_NULL};
  CommState state{CommState::START};
};

#include "communicator.inl"

#endif
