#ifndef INCLUDED_COMMSTATE_HPP
#define INCLUDED_COMMSTATE_HPP

#include <cstdint>

/**
   CommState
**/

enum class CommState : uint8_t {
  START = 0,
  INCOMING_CENTERS = 1,
  BUCKETING_OTHERS = 2,
  SIZE = 3,
  SIZE_FINISHED = 4,
  OUTGOING_BUCKETS = 5,
  INCOMING_BUCKETS = 6,
  SIEVING = 7,
  INSERTION_SIZES = 8,
  INSERTION_EXCHANGE = 9,
  INSERTING = 10,
};

#endif
