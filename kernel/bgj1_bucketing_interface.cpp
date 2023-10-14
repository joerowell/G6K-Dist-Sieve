#include "bgj1_bucketing_interface.hpp"
#include "layouts.hpp"
#include "siever.h"
#include <numeric>

bgj1_bucketing_interface::bgj1_bucketing_interface(
    const uint64_t buckets_per_rank, const unsigned nr_ranks,
    const unsigned batches, MPI_Comm comm) noexcept {

  comms_.resize(batches);
  for (auto &comm_ : comms_) {
    comm_.duplicate_comm(comm);
  }

  ops_ = std::vector<padded_atom_unsigned>(batches);
  states_.resize(batches, 0);
  counts_.resize(batches, 0);

  sync_headers_.resize(batches);

  // The sizes_ vector is rather straightforward: each process has
  // buckets_per_rank elements and we have batches of them.
  sizes_.resize(batches);
  for (auto &v : sizes_) {
    v.resize(buckets_per_rank * nr_ranks);
  }
  bucket_positions_.resize(buckets_per_rank * nr_ranks);
}

void bgj1_bucketing_interface::setup_bucket_positions(
    const size_t size, const unsigned bucket_batches,
    const uint64_t buckets_per_rank, const unsigned nr_ranks,
    const unsigned rank) noexcept {

  // This function essentially takes "size" and offsets each rank's
  // bucket storage by the number of buckets it will send.
  // Since we know this ahead of time (it's some multiple of the number of
  // centers) this allows us to avoid some collective communications during
  // sieving, which lowers the bandwidth cost somewhat (the bandwidth cost isn't
  // much: it's more that the system calls are expensive). In other words: when
  // this function returns, bucket_positions will look like this: [size, size +
  // buckets_per_rank[0], size + buckets_per_rank[0] + buckets_per_rank[1], ...]
  // NOTE: to prevent extra allocations at the database level, we skip our rank
  // here: we do not include our own rank in the computation.
  // NOTE: the scale factor of "bucket_batches" means that each bucket has the
  // appropriate amount of storage for all of the centers that will be in
  // flight.

  uint64_t bucket_count{size};
  for (unsigned i = 0; i < nr_ranks; i++) {
    if (i == rank)
      continue;
    for (unsigned j = 0; j < bucket_batches; j++) {
      bucket_positions_[i * bucket_batches + j] = bucket_count;
      bucket_count += buckets_per_rank;
    }
  }
}

void bgj1_bucketing_interface::setup_sync_objects(
    const unsigned size) noexcept {
  for (auto &v : sync_headers_) {
    v.resize(size);
  }
}
