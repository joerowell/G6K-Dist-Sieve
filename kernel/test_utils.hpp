#ifndef INCLUDED_TESTUTILS_HPP
#define INCLUDED_TESTUTILS_HPP

/**
   TestUtils. This file just contains some useful code definitions etc for
various parts of the code base. These are all injected into the global
namespace.
**/

#ifndef DOCTEST_CONFIG_DISABLE
/**
   throw_if_testing. If this is false, then any function annotated with
   noexcept(noexcept(throw_if_testing)) (alternatively, MPI_DIST_MAY_THROW)
   may throw during the lifetime of the program. This will only happen if
 MPI_DIST_TEST is set.
 **/
static constexpr bool throw_if_testing = false;

/**
   MPI_DIST_ARE_TESTING. This is true if MPI_DIST_TEST is set and false
otherwise. This is essentially always the opposite of throw_if_testing: this is
to get around some awkwardness with conditional noexcept.
**/
static constexpr bool MPI_DIST_ARE_TESTING = true;
#else
/**
   throw_if_testing. If this is true, then any function annotated with
   noexcept(noexcept(throw_if_testing)) (alternatively, MPI_DIST_MAY_THROW)
   will not throw during the lifetime of the program. This will only happen if
 MPI_DIST_TEST is not set.
 **/
static constexpr bool throw_if_testing = true;
/**
   MPI_DIST_ARE_TESTING. This is true if MPI_DIST_TEST is set and false
otherwise. This is essentially always the opposite of throw_if_testing: this is
to get around some awkwardness with conditional noexcept.
**/
static constexpr bool MPI_DIST_ARE_TESTING = false;
#endif

/**
   MPI_DIST_MAY_THROW. This macro denotes that a function may throw under
   testing conditions. This primarily exists to make it so that we get the
benefits of noexcept code in practice, but also allows us to check certain
principles in testing.
**/
#define MPI_DIST_MAY_THROW noexcept(throw_if_testing)

#ifdef G6K_MPI
// THROW_OR_OPTIMISE. This macro checks whether MPI_DIST_ARE_TESTING is set:
// if so, then the function is executed and, if true, then an exception (of type
// exception_type) is thrown with a message of message.
// If MPI_DIST_ARE_TESTING is not set, then this macro informs the compiler that
// it can optimise away for function being false. This may have a benefit when
// it comes to certain inlining decisions later.
#define THROW_OR_OPTIMISE(function, exception_type, message)                   \
  do {                                                                         \
    if constexpr (MPI_DIST_ARE_TESTING) {                                      \
      if (function) {                                                          \
        throw exception_type(message);                                         \
      }                                                                        \
    } else {                                                                   \
      if (function) {                                                          \
        __builtin_unreachable();                                               \
      }                                                                        \
    }                                                                          \
  } while (0)
#else
// This is the empty version of the above.
#define THROW_OR_OPTIMISE(function, exception_type, message)                   \
  do {                                                                         \
  } while (0)
#endif
#endif
