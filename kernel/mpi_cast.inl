#ifndef INCLUDED_MPI_CAST_HPP
#error Do not include mpi_cast.inl without mpi_cast.hpp
#endif

inline MPI_Comm MPI_Cast::uint64_to_mpi_comm(const uint64_t in) noexcept {
  return (MPI_Comm)(in);
}

inline uint64_t MPI_Cast::mpi_comm_to_uint64(const MPI_Comm in) noexcept {
  return (uint64_t)(in);
}
