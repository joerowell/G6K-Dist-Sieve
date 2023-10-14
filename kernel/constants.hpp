#ifndef INCLUDED_CONSTANTS_HPP
#define INCLUDED_CONSTANTS_HPP

#include "g6k_config.h"
#include <array>
#include <cstdint>

// Maximum dimension of the local blocks we sieve in.
#ifndef MAX_SIEVING_DIM
#define MAX_SIEVING_DIM 128
#endif

#ifndef XPC_THRESHOLD
#define XPC_THRESHOLD 96
#endif

#ifndef XPC_BUCKET_THRESHOLD
#define XPC_BUCKET_THRESHOLD 102
#endif

#ifndef OTF_LIFT_HELPER_DIM
#define OTF_LIFT_HELPER_DIM 16
#endif

static constexpr unsigned int XPC_WORD_LEN =
    4; // number of 64-bit words of simhashes
static constexpr unsigned int XPC_BIT_LEN = 256; // number of bits for simhashes
static constexpr unsigned int XPC_SAMPLING_THRESHOLD =
    96; // XPC Threshold for partial sieving while sampling
static constexpr unsigned int XPC_THRESHOLD_TRIPLE =
    97; // XPC Threshold for triple sieve
static constexpr unsigned int XPC_THRESHOLD_TRIPLE_INNER_CHECK =
    133; // XPC Threshold for triple sieve in the inner-loop
static constexpr float X1X2 =
    0.108; // Threshold to put vector in filtered list ~(1/3)^2

static constexpr unsigned int MIN_ENTRY_PER_THREAD =
    100; // factor that determines minimum size of work batch to distribute to a
         // thread.

#define REDUCE_LEN_MARGIN                                                      \
  1.01 // Minimal improvement ratio to trigger a reduction
       // (make sure of worthy progress, avoid infinite loops due to
       // numerical-errors)

#define REDUCE_LEN_MARGIN_HALF                                                 \
  1.005 // Minimal improvement ratio to validate a reduction

#define CACHE_BLOCK                                                            \
  512 // Local loops length for cache-friendlyness. Note that triple_sieve_mt
      // has its separate variable for that.

#define VERBOSE false

typedef float LFT;  // Low Precision floating points for vectors yr
typedef double FT;  // High precision floating points for vectors y
typedef int16_t ZT; // Integer for vectors x (i.e. coefficients of found vectors
                    // wrt the given basis)
typedef uint32_t IT; // Index type for indexing into the main database of
                     // vectors (32 bits for now, limiting db_size to 2^32-1)

typedef std::array<uint64_t, XPC_WORD_LEN>
    CompressedVector; // Compressed vector type for XOR-POPCNT

// made a typedef to make it better customizable. (If we increase the size of IT
// from 32 bits, we want to increase this accordingly).

using SimHashDescriptor =
    unsigned[XPC_BIT_LEN][6]; // typedef to avoid some awkward declarations.

using UidType =
    uint64_t; // Type of the hash we are using for collision detection.

using UidCoeffType = uint64_t;

#define DB_UID_SPLIT 8191

// This struct is used inside BDGL / MPI for transferring over reduction
// information.

struct Reduction {
  UidType uid;
  FT len;
  IT pos;
};

#endif
