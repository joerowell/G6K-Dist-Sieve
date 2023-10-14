#include "siever_types.hpp"
#include <type_traits>

// This static assertion is here just to make sure that everything lines up as
// expected. Functions that use this behaviour should also assert to it, just so
// all is good collectively.
static_assert(
    std::is_same_v<std::underlying_type_t<DistSieverType>, unsigned>,
    "Error: DistSieverType no longer has an underlying type of unsigned");
