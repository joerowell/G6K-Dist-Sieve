#ifndef INCLUDED_COMMUNICATOR_HPP
#error Do not include communicator.inl without communicator.hpp
#endif

inline void Communicator::duplicate_comm(MPI_Comm comm) noexcept {
  MPI_Comm_dup(comm, &this->comm);
  MPI_Comm_set_errhandler(this->comm, MPI_ERRORS_RETURN);
}

inline MPI_Comm &Communicator::get_comm() noexcept { return comm; }

inline void Communicator::start_incoming_centers() noexcept {
  state = CommState::INCOMING_CENTERS;
}

inline bool Communicator::has_incoming_centers() const noexcept {
  return state == CommState::INCOMING_CENTERS;
}

inline void Communicator::start_bucketing() noexcept {
  state = CommState::BUCKETING_OTHERS;
}

inline bool Communicator::is_bucketing() const noexcept {
  return state == CommState::BUCKETING_OTHERS;
}

inline void Communicator::start_size() noexcept { state = CommState::SIZE; }

inline bool Communicator::is_sizing() const noexcept {
  return state == CommState::SIZE;
}

inline void Communicator::mark_size_finished() noexcept {
  state = CommState::SIZE_FINISHED;
}

inline bool Communicator::has_finished_size() const noexcept {
  return state == CommState::SIZE_FINISHED;
}

inline void Communicator::start_incoming_buckets() noexcept {
  state = CommState::INCOMING_BUCKETS;
}

inline bool Communicator::is_receiving_buckets() const noexcept {
  return state == CommState::INCOMING_BUCKETS;
}

inline void Communicator::start_outgoing_buckets() noexcept {
  state = CommState::OUTGOING_BUCKETS;
}

inline bool Communicator::is_sending_buckets() const noexcept {
  return state == CommState::OUTGOING_BUCKETS;
}

inline void Communicator::reset() noexcept { state = CommState::START; }

inline bool Communicator::is_reset() const noexcept {
  return state == CommState::START;
}

inline CommState Communicator::get_state() const noexcept { return state; }

inline void Communicator::start_sieving() noexcept {
  state = CommState::SIEVING;
}

inline bool Communicator::is_sieving() const noexcept {
  return state == CommState::SIEVING;
}

inline void Communicator::free_comm() noexcept { MPI_Comm_free(&this->comm); }

inline void Communicator::start_incoming_insertion_size() noexcept {
  state = CommState::INSERTION_SIZES;
}

inline bool Communicator::is_receiving_insertion_sizes() const noexcept {
  return state == CommState::INSERTION_SIZES;
}

inline void Communicator::start_insertion_exchange() noexcept {
  state = CommState::INSERTION_EXCHANGE;
}

inline bool Communicator::is_receiving_insertions() const noexcept {
  return state == CommState::INSERTION_EXCHANGE;
}

inline void Communicator::start_insertions() noexcept {
  state = CommState::INSERTING;
}

inline bool Communicator::is_inserting() const noexcept {
  return state == CommState::INSERTING;
}
