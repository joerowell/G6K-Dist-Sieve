#ifndef INCLUDED_LAYOUTS_HPP
#error Do not include Layouts.hpp without Layouts.inl
#endif

template <typename T> constexpr bool Layouts::is_fundamental_type() noexcept {
  // List comes from
  // https://www.mpi-forum.org/docs/mpi-2.2/mpi22-report/node44.htm
  using RT =
      std::remove_const_t<std::remove_reference_t<std::remove_const_t<T>>>;
  return std::is_same_v<RT, char> || std::is_same_v<RT, signed short int> ||
         std::is_same_v<RT, signed int> ||
         std::is_same_v<RT, signed long int> ||
         std::is_same_v<RT, signed long long int> ||
         std::is_same_v<RT, signed char> || std::is_same_v<RT, unsigned char> ||
         std::is_same_v<RT, unsigned short int> ||
         std::is_same_v<RT, unsigned int> ||
         std::is_same_v<RT, unsigned long int> ||
         std::is_same_v<RT, unsigned long long int> ||
         std::is_same_v<RT, float> || std::is_same_v<RT, double> ||
         std::is_same_v<RT, long double> || std::is_same_v<RT, int8_t> ||
         std::is_same_v<RT, int16_t> || std::is_same_v<RT, int32_t> ||
         std::is_same_v<RT, int64_t> || std::is_same_v<RT, uint8_t> ||
         std::is_same_v<RT, uint16_t> || std::is_same_v<RT, uint32_t> ||
         std::is_same_v<RT, uint64_t> || std::is_same_v<RT, bool>;
}

template <typename T>
constexpr inline MPI_Datatype Layouts::get_data_type() noexcept {
  // We actually need to act on the fundamental type that underlies `T`, because
  // `T` could be (for example) a int&, but for the sake of MPI we need to get
  // the type for `int`.
  // N.B It's possible that this comparison could be skipped by using
  // std::decay, which would also allow us to treat arrays etc in this manner.
  using RT =
      std::remove_const_t<std::remove_reference_t<std::remove_const_t<T>>>;
  static_assert(Layouts::is_fundamental_type<RT>(),
                "Error: cannot instantiate Layouts::get_data_type on "
                "non-fundamental type T");
  // List comes from
  // https://www.mpi-forum.org/docs/mpi-2.2/mpi22-report/node44.htm
  // We ignore the complex types, because C's complex type is slightly
  // different from C++'s (and we don't use it anyway).

  if constexpr (std::is_same_v<RT, char>) {
    return MPI_CHAR;
  } else if constexpr (std::is_same_v<RT, signed short int>) {
    return MPI_SHORT;
  } else if constexpr (std::is_same_v<RT, signed int>) {
    return MPI_INT;
  } else if constexpr (std::is_same_v<RT, signed long int>) {
    return MPI_LONG;
  } else if constexpr (std::is_same_v<RT, signed long long int>) {
    return MPI_LONG_LONG_INT;
  } else if constexpr (std::is_same_v<RT, signed char>) {
    return MPI_SIGNED_CHAR;
  } else if constexpr (std::is_same_v<RT, unsigned char>) {
    return MPI_UNSIGNED_CHAR;
  } else if constexpr (std::is_same_v<RT, unsigned short int>) {
    return MPI_UNSIGNED_SHORT;
  } else if constexpr (std::is_same_v<RT, unsigned int>) {
    return MPI_UNSIGNED;
  } else if constexpr (std::is_same_v<RT, unsigned long int>) {
    return MPI_UNSIGNED_LONG;
  } else if constexpr (std::is_same_v<RT, unsigned long long int>) {
    return MPI_UNSIGNED_LONG_LONG;
  } else if constexpr (std::is_same_v<RT, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<RT, double>) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same_v<RT, long double>) {
    return MPI_LONG_DOUBLE;
  } else if constexpr (std::is_same_v<RT, int8_t>) {
    return MPI_INT8_T;
  } else if constexpr (std::is_same_v<RT, int16_t>) {
    return MPI_INT16_T;
  } else if constexpr (std::is_same_v<RT, int32_t>) {
    return MPI_INT32_T;
  } else if constexpr (std::is_same_v<RT, int64_t>) {
    return MPI_INT64_T;
  } else if constexpr (std::is_same_v<RT, uint8_t>) {
    return MPI_UINT8_T;
  } else if constexpr (std::is_same_v<RT, uint16_t>) {
    return MPI_UINT16_T;
  } else if constexpr (std::is_same_v<RT, uint32_t>) {
    return MPI_UINT32_T;
  } else if constexpr (std::is_same_v<RT, uint64_t>) {
    return MPI_UINT64_T;
  } else if constexpr (std::is_same_v<RT, bool>) {
    return MPI_C_BOOL;
  }

  __builtin_unreachable();
}
