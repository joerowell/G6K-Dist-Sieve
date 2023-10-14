#ifndef INCLUDED_CONTEXTCHANGE_HPP
#define INCLUDED_CONTEXTCHANGE_HPP

#include <iosfwd> // Needed for ostream forward declaration.
/**
   ContextChange. This enum contains the various packaging formats for
   the change of context messages that are used inside G6K.

   This enum exists to allow the caller to pack the message type (EL, SL, ER)
etc into the upper 32 bits and the parameter into the lower 32 bits. This makes
packing the operations efficient.
**/

enum class ContextChange : unsigned {
  /**
   STOP. This corresponds to a message sent for terminating the sieve.
  **/
  STOP = 0,

  /**
     EL. This corresponds to an extend-left message.
  **/
  EL = 1,
  /**
     ER. This corresponds to an extend-right message.
  **/
  ER = 2,
  /**
     SL. This corresponds to a shrink-left message.
  **/
  SL_REDIST = 3,

  /**
     IL. This corresponds to an initialize local message.
  **/
  IL = 4,

  /**
     GROW_SMALL. This corresponds to a grow message with a large parameter = 0.
  **/
  GROW_SMALL = 5,

  /**
     GROW_LARGE. This corresponds to a grow message with a large parameter > 0
  **/
  GROW_LARGE = 6,

  /**
     SHRINK. This corresponds to a shrink message.
  **/
  SHRINK = 7,

  /**
     SORT. This corresponds to a sort message.
  **/
  SORT = 8,

  /**
     CHANGE_STATUS. This corresponds to a change status message.
  **/
  CHANGE_STATUS = 9,

  /**
     GSO_PP. This corresponds to a message sent for GSO postprocessing.
  **/
  GSO_PP = 10,

  /**
     LOAD_GSO. This corresponds to a message sent for loading the GSO.
  **/
  LOAD_GSO = 11,

  /**
     HISTO. This corresponds to a message sent for compiling the histogram.
  **/
  HISTO = 12,

  /**
     BGJ1. This corresponds to telling attached ranks to run a BGJ1 sieve.
  **/
  BGJ1 = 13,

  /**
     ORDERED_DB_SPLIT. This corresponds to a message that splits the database.
  **/
  ORDERED_DB_SPLIT = 14,

  /**
     BUCKET. This message corresponds to a non-bdgl bucket.
  **/
  BUCKET = 15,

  /**
     DB_SIZE. This message corresponds to a request for the database size.
  **/
  DB_SIZE = 16,

  /**
     RESET_STATS. This message corresponds to a request to clear the stats
  counting.
  **/
  RESET_STATS = 17,

  /**
     DB_SPLIT_UIDS.
  **/
  DB_SPLIT_UIDS = 18,

  /**
     INSERT.
  **/
  INSERT = 19,

  /**
     ADHOC_INSERT.
  **/
  ADHOC_INSERT = 20,
  STOP_SIEVING = 21,
  STATS_LONG = 22,
  STATS_DOUBLE = 23,
  DUP_INSERT = 24,
  BUCKET_SIZES = 25,
  BUCKET_START = 26,
  REDIST_DB = 27,
  RESERVE = 28,
  GET_CPU_TIME = 29,
  GET_EXTRA_MEMORY = 30,
  RECONCILE_DB = 31,
  GET_GLOBAL_SATURATION = 32,
  CHANGE_ROOT_RANK = 33,
  COLLECT_MEMORY = 34,
  GATHER_BUCKETS = 35,
  SEND_TOPOLOGY = 36,
  BROADCAST_PARAMS = 37,
  BEST_LIFTS = 38,
  GET_DB = 39,
  SPLIT_DB = 40,
  GET_GLOBAL_VARIANCE = 41,
  GET_MIN_MAX_LEN = 42,
  LIFT_BOUNDS = 43,
  BGJ1_BUCKET_SEND = 44,
  BGJ1_CENTER_SEND = 45,
  BGJ1_SEND_SIZE = 46,
  BGJ1_PROCESS_STOP = 47,
  MESSAGES_FOR = 48,
  BANDWIDTH_FOR = 49,
  TOTAL_MESSAGES = 50,
  TOTAL_BANDWIDTH = 51,
  IS_ONE_G6K = 52,
  CHECK_FINAL_INSERTIONS = 53,
  SL_NO_REDIST = 54,
  GET_UNIQUE_RATIO = 55,
  GET_ADJUST_TIMINGS = 56,
  RESET_BANDWIDTH = 57,

  BDGL = 58,
  BDGL_BUCKET = 59,
  BDGL_QUERY = 60,
  BDGL_REMOVE_DUP = 61,
  LAST = 62,
};

/**
   ==. This function returns true if `lhs` == `rhs`. This function never throws.
   @param[in] lhs: the left hand side.
   @param[in] rhs: the right hand side.
   @return lhs == rhs.
**/
constexpr bool operator==(const unsigned lhs, const ContextChange rhs) noexcept;
/**
   ==. This function returns true if `lhs` == `rhs`. This function never throws.
   @param[in] lhs: the left hand side.
   @param[in] rhs: the right hand side.
   @return lhs == rhs.
**/
constexpr bool operator==(const ContextChange lhs, const unsigned rhs) noexcept;
/**
   ==. This function returns true if `lhs` != `rhs`. This function never throws.
   @param[in] lhs: the left hand side.
   @param[in] rhs: the right hand side.
   @return lhs != rhs.
**/
constexpr bool operator!=(const unsigned lhs, const ContextChange rhs) noexcept;
/**
   !=. This function returns true if `lhs` != `rhs`. This function never throws.
   @param[in] lhs: the left hand side.
   @param[in] rhs: the right hand side.
   @return lhs != rhs.
**/
constexpr bool operator!=(const ContextChange lhs, const unsigned rhs) noexcept;

std::ostream &operator<<(std::ostream &os, const ContextChange cc) noexcept;

// Inline defs live here.
#include "context_change.inl"

#endif
