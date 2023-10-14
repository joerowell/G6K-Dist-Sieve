#ifndef INCLUDED_LAYOUTS_HPP
#define INCLUDED_LAYOUTS_HPP

#include "mpi.h" // Needed for MPI things.
#include <cstdint>
#include <vector>       // Needed for API.
struct CompressedEntry; // Needed for API.
struct Entry;           // Needed for API.

using CEIter = std::vector<CompressedEntry>::const_iterator;
using IntIter = std::vector<unsigned>::const_iterator;

/**
   Layouts. This namespace is a procedural wrapper for gaining access to
serialisation types. This namespace exists solely to have all of the
serialisation creation types in one place.

This namespace provides a series of named functions that retrieve MPI_Datatypes.
You can retrieve these directly:

\code{.cpp}
   Layouts::get_param_type();
\endcode{}

@remarks Please be careful if you add code to this namespace! It turns out that
MPI sometimes exhibits difficult to debug behaviour when it comes to producing
new data types. For example, the get_cc_layout (at the time of writing) uses
MPI_Type_contiguous: before we used MPI_Type_vector and ended up with a weird
case where ASAN builds passed, but non-ASAN builds failed due to different
codegen (no memory leaks were introduced -- the tests just failed). This took
some hours to debug: so, please be careful, and write tests!

@remarks Note that this namespace does not always include serialisation types
for all data that is used in the sieve. If there is not a serialisation routine
for the type of choice (e.g SieveStatistics) then this typically is because the
class itself needs to provide serialisation logic.
**/

namespace Layouts {
/**
   get_param_layout. This function returns a new MPI
Datatype used to represent a SieverParams object. This function does not throw.
@return a MPI datatype used to serialise a SieverParams object.
**/
MPI_Datatype get_param_layout() noexcept;

/**
   get_cc_layout. This function returns a copy of the handle to the MPI Dataype
   used to represent a context change message. This function does not throw.
   @return a copy of the handle used to send context change messages.
   @remarks This function is currently unused: we simply hardcode the size as a
two element array.
**/
MPI_Datatype get_cc_layout() noexcept;

/**
   get_entry_layout. This function produces a new MPI_Datatype that allows one
 to send an Entry to another rank. This function does not throw.

 @param[in] n: the number of elements to send from a given entry.
   @remarks To be able to send a std::vector of entries, call
 get_entry_vector_layout instead. This is because the layout isn't naturally
 contiguous, and thus the sending and receiving can be tricky to get right.
 @return a datatype for sending a single entry to another rank.
 @remarks The exact type that is returned here depends on the compilation flags.
 By default the layout returned here is a layout that serialises only e.x[0] ->
e.x[n] for some Entry e. However, layouts that serialise e.x and e.c can also be
chosen, along with layouts that serialise e.x, e.c, e.yr and e.len. See
Siever::recompute_recv() for more.
**/
MPI_Datatype get_entry_layout(const unsigned n) noexcept;

uint64_t get_entry_size(const unsigned n) noexcept;

/**
     get_entry_vector_layout. This function produces a new MPI_Datatype that
allows one to send a vector of entries to another rank. This function does not
throw.
 @param[in] n: the number of elements to send from a given entry.
 @return a datatype for sending a contiguous set of entries.
 @remarks As with get_entry_layout, the exact type returned here depends on the
 flags set during compilation.
**/
MPI_Datatype get_entry_vector_layout(const unsigned n) noexcept;

/**
   get_entry_vector_layout_x_only. This function returns an MPI_Datatype that
allows one to send a vector of entries to another rank. This function returns a
datatype that only ever serialises the `x` representation.
    @param[in] n: the number of elements to send from a given entry.
    @return a datatype for sending a contiguous set of entries via their x
representation.
**/
MPI_Datatype get_entry_vector_layout_x_only(const unsigned n) noexcept;

/**
   get_entry_layout_non_contiguous. This function produces a new MPI_Datatype
   that allows the caller to send a non-contiguous set of entries to another
rank.

   @param[in] begin: the (inclusive) entry to start sending in the cdb.
   @param[in] end: the (exclusive) entry to stop sending in the cdb.
   @param[in] n: the number of elements to send from a given entry.
   @return a datatype for sending a non-contiguous portion of the database.
   @remarks As with get_entry_layout, the exact type returned here depends on
the flags set during compilation.
**/
MPI_Datatype get_entry_layout_non_contiguous(const CEIter begin,
                                             const CEIter end,
                                             const unsigned n) noexcept;

MPI_Datatype get_sync_header_type() noexcept;

MPI_Datatype get_entry_type(const unsigned n) noexcept;

/**
   get_entry_layout_non_contiguous. This function produces a new MPI_Datatype
   that allows the caller to send a non-contiguous set of entries to another
rank.

   @param[in] begin: the (inclusive) entry to start sending in the cdb.
   @param[in] end: the (exclusive) entry to stop sending in the cdb.
   @param[in] n: the number of elements to send from a given entry.
   @return a datatype for sending a non-contiguous portion of the database.
   @remarks As with get_entry_layout, the exact type returned here depends on
the flags set during compilation.
**/
MPI_Datatype get_entry_layout_non_contiguous(const IntIter begin,
                                             const IntIter end,
                                             const unsigned n) noexcept;

MPI_Datatype get_entry_layout_from_cdb(const std::vector<CompressedEntry> &cdb,
                                       const IntIter begin, const IntIter end,
                                       const unsigned n,
                                       std::vector<int> &offsets) noexcept;

MPI_Datatype get_entry_layout_from_cdb_x_only(
    const std::vector<CompressedEntry> &cdb, const IntIter begin,
    const IntIter end, const unsigned n, std::vector<int> &offsets) noexcept;

MPI_Datatype get_entry_layout_non_contiguous(const CEIter begin,
                                             const CEIter end,
                                             const MPI_Datatype type) noexcept;

MPI_Datatype get_product_lsh_data_type_aes() noexcept;

struct ProductLSHLayout {
  size_t n, blocks;
  int64_t code_size;
  unsigned multi_hash;
  int64_t seed;
};

/**
   get_data_type. This function returns the fundamental MPI Data type for
sending objects of type T. This only works if instantiated with a type that MPI
can natively send (e.g those that are listed with fundamental types in the MPI
standard. If T does not correspond to such a type then this function will not
compile.

   @tparam T: the type to serialise.
   @return a datatype that can be used to serialise objects of type T.
**/
template <typename T> constexpr inline MPI_Datatype get_data_type() noexcept;

/**
   is_fundamental_type. This function returns true if T is a fundamental type
 and false otherwise. If this function returns true for T then you can retrieve
 the data type needed to serialise T using get_data_type().
   @tparam T the type to serialise.
   @return true if T is a fundamental type and false otherwise.
 **/
template <typename T> constexpr bool is_fundamental_type() noexcept;
} // namespace Layouts

// Inline definitions live here.
#include "layouts.inl"

#endif
