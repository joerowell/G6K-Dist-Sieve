#include "mpi_wrapper.hpp"
#ifdef MPI_DIST_TEST
#include "doctest/extensions/doctest_mpi.h"
#endif
#include "layouts.hpp"
#include <algorithm>
#include <cassert>
#ifdef MPI_DIST_TEST
#include <stdexcept> // Needed for exceptions, but only during debugging.
#endif

#include "context_change.hpp" // Needed for tagging.
#include "siever.h"           // Needed for almost everything.
#include "statistics.hpp"
#include "thread_pool.hpp"
#include <cmath>
#include <unordered_map>

// Although we always have this global, it's only conditionally used.
#ifdef MPI_TRACK_BANDWIDTH
MPIBandwidthTracker tracker;
#endif

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("can set root rank to 0", 3) {
  SUBCASE("root already has rank of 0") {
    // This manifests as set_root_rank_to_global_root_rank returning the
    // existing communicator.
    constexpr auto root_rank = 0;
    auto new_comm = MPIWrapper::set_root_rank_to_global_root_rank(
        test_rank, test_rank == root_rank, test_comm);
    CHECK(new_comm == test_comm);
    // No need to free here: it's freed elsewhere.
  }

  SUBCASE("normal operation") {
    // Here the root rank is 1, so we can see a swap between these ranks.
    constexpr auto root_rank = 1;
    auto new_comm = MPIWrapper::set_root_rank_to_global_root_rank(
        test_rank, (test_rank == root_rank), test_comm);
    int new_rank;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_CHECK(0, new_rank == 1);
    MPI_CHECK(1, new_rank == 0);
    // The third rank has the same rank as before.
    MPI_CHECK(2, new_rank == test_rank);
    // Need to free this one, as it's a new comm.
    MPI_Comm_free(&new_comm);
  }
}
#endif

MPI_Comm MPIWrapper::set_root_rank_to_global_root_rank(const int rank,
                                                       const bool is_root,
                                                       MPI_Comm comm) noexcept {

  CPUCOUNT(700);
  // Collect the root rank of the existing ranks.
  int root_rank = (is_root) ? rank : 0;

  TRACK(ContextChange::CHANGE_ROOT_RANK, sizeof(root_rank));
  MPI_Allreduce(MPI_IN_PLACE, &root_rank, 1, MPI_INT, MPI_MAX, comm);
  // If we've already got a root rank that's got the right rank, bail.
  if (root_rank == MPIWrapper::global_root_rank) {
    return comm;
  }

  // Otherwise, we need to switch the ranks so that the rank of the root is is
  // MPIWrapper::global_root_rank.
  int new_rank;
  if (is_root) {
    new_rank = MPIWrapper::global_root_rank;
  } else if (rank == MPIWrapper::global_root_rank) {
    new_rank = root_rank;
  } else {
    // If not involved, we'll just keep the old rank.
    new_rank = rank;
  }

  // Now we have to create a new communicator, so that the rank is updated.
  MPI_Comm new_comm;
  MPI_Comm_split(comm, 0, new_rank, &new_comm);
  return new_comm;
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("can collect memory", 2) {
  std::vector<uint64_t> memory;
  constexpr std::array<uint64_t, 2> mems{5, 10};
  MPIWrapper::collect_memory(memory, mems[test_rank], test_comm);
  // Check that it adds up and the results are in order.
  // Note: the output array also contains the memory associated with _this_
  // rank!
  REQUIRE(memory.size() == 2);
  CHECK(memory[0] == mems[0]);
  CHECK(memory[1] == mems[1]);
}
#endif

void MPIWrapper::collect_memory(std::vector<uint64_t> &memory_per_rank,
                                const uint64_t memory, MPI_Comm comm) noexcept {
  CPUCOUNT(701);
  // Get the number of ranks first and resize `memory_per_rank` to hold the
  // results.
  int size;
  int rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  memory_per_rank.resize(static_cast<unsigned>(size));

  // This allows us to use MPI_IN_PLACE (saves us an outgoing message).
  memory_per_rank[static_cast<unsigned>(rank)] = memory;
  // Two things of note here:
  // 1. You might look at this and think "how is this guaranteed to put the
  // results
  //    in rank (i.e rank) order?" That's a good question: it turns out the MPI
  //    standard dictates it. See
  //    https://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf Page 150,
  //    line 10. As a result, it turns out this is all we need.
  // 2. The second "1" means "we expect one element per rank". This is specified
  //    in the description for MPI_Gather
  //    (https://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf, Section 5.5,
  //    recvcount). This can be confusing if you haven't seen it before.
  TRACK(ContextChange::COLLECT_MEMORY, sizeof(memory));
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_UINT64_T, memory_per_rank.data(), 1,
                MPI_UINT64_T, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("gather_buckets", 2) {
  const std::vector<uint64_t> ref{20, 36};
  std::vector<uint64_t> results;
  MPIWrapper::gather_buckets(ref[static_cast<unsigned>(test_rank)], test_rank,
                             results, test_comm);
  CHECK(results == ref);
}
#endif

void MPIWrapper::gather_buckets(uint64_t &buckets, unsigned &scale_factor,
                                unsigned &bucket_batches,
                                unsigned &scratch_buffers,
                                MPI_Comm comm) noexcept {
  CPUCOUNT(703);
  int nr_ranks, rank;
  MPI_Comm_size(comm, &nr_ranks);
  MPI_Comm_rank(comm, &rank);

  if (rank == MPIWrapper::global_root_rank) {
    // Cap scratch_buffers to be at most bucket_batches.
    if (scratch_buffers > bucket_batches) {
      scratch_buffers = bucket_batches;
    }
  }

  std::array<uint64_t, 4> params{buckets};
  if (rank == MPIWrapper::global_root_rank) {
    params[1] = scale_factor;
    params[2] = bucket_batches;
    params[3] = scratch_buffers;
  }

  // N.B This works because all non-root ranks supply 0s for the
  // (scale_factor, bucket_batches, scratch_buffers) params, and thus
  // the max is just the value from the root.
  MPI_Allreduce(MPI_IN_PLACE, &params, 4, MPI_UINT64_T, MPI_MAX, comm);

  TRACK(ContextChange::GATHER_BUCKETS, sizeof(params));

  buckets = params[0] * params[1];
  scale_factor = params[1];
  bucket_batches = params[2];
  scratch_buffers = params[3];
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("can send and receive topology", 2) {
  auto topology = DistSieverType::ShuffleSiever;
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::send_topology(topology, test_comm);
  } else {
    DistSieverType out;
    MPIWrapper::get_topology(out, test_comm);
    CHECK(out == topology);
  }
}
#endif

void MPIWrapper::send_topology(DistSieverType topology,
                               MPI_Comm comm) noexcept {
  CPUCOUNT(704);
  // Just to check the right sender type is used.
  static_assert(
      std::is_same_v<std::underlying_type_t<DistSieverType>, unsigned>,
      "Error: DistSieverType no longer has an underlying type of unsigned");
  // Note: this is an explicit type conversion because the C++ standard is
  // actually a bit unclear on if it's legal to reinterpret_cast topology to a
  // unsigned*. This is just for safety across platforms.
  unsigned topology_as_unsigned = static_cast<unsigned>(topology);

  // The MPI_IN_PLACE just says "store the result in the top_as_unsigned
  // argument". Note: this is an Allreduce because all ranks need to know the
  // output.
  TRACK(ContextChange::SEND_TOPOLOGY, sizeof(topology_as_unsigned));
  MPI_Allreduce(MPI_IN_PLACE, &topology_as_unsigned, 1, MPI_UNSIGNED, MPI_MAX,
                comm);
}

void MPIWrapper::get_topology(DistSieverType &topology,
                              MPI_Comm comm) noexcept {
  CPUCOUNT(705);
  // Just to check the right sender type is used.
  static_assert(
      std::is_same_v<std::underlying_type_t<DistSieverType>, unsigned>,
      "Error: DistSieverType no longer has an underlying type of unsigned");

  // N.B similarly to with send_topology to prevent strange type conversion
  // errors we read into a standardly defined type and then unpack.
  auto as_unsigned{static_cast<unsigned>(DistSieverType::Null)};
  MPI_Allreduce(MPI_IN_PLACE, &as_unsigned, 1, MPI_UNSIGNED, MPI_MAX, comm);
  // By send_topology's signature it would already be UB to have anything passed
  // in that isn't a DistSieverType,s o this is legal.
  topology = static_cast<DistSieverType>(as_unsigned);
}

template <bool track_count = false>
static void broadcast_context_change(const ContextChange operation,
                                     const unsigned p, MPI_Comm comm) noexcept {
  CPUCOUNT(706);
  std::array<unsigned, 2> packed{static_cast<unsigned>(operation), p};
  MPI_Bcast(packed.data(), 2, MPI_UNSIGNED, MPIWrapper::global_root_rank, comm);
  if (track_count) {
    TRACK(operation, sizeof(packed));
  } else {
    TRACK_NO_COUNT(operation, sizeof(packed));
  }
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("broadcast_context_change", 2) {
  std::array<ContextChange, 3> change{ContextChange::EL, ContextChange::ER,
                                      ContextChange::SL};
  std::array<unsigned, 3> params{1, 2, 3};

  for (unsigned i = 0; i < 3; i++) {
    if (test_rank == MPIWrapper::global_root_rank) {
      broadcast_context_change(change[i], params[i], test_comm);
    } else {
      std::array<unsigned, 2> arr;
      MPI_Bcast(arr.data(), 2, MPI_UNSIGNED, MPIWrapper::global_root_rank,
                test_comm);
      CHECK(arr[0] == change[i]);
      CHECK(arr[1] == params[i]);
    }
  }
}
#endif

static std::array<unsigned, 2> receive_context_change(MPI_Comm comm) noexcept {
  CPUCOUNT(707);
  std::array<unsigned, 2> arr;
  MPI_Bcast(arr.data(), 2, MPI_UNSIGNED, MPIWrapper::global_root_rank, comm);
  return arr;
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("receive_cc", 2) {
  constexpr std::array<ContextChange, 3> change{
      ContextChange::EL, ContextChange::ER, ContextChange::SL};
  constexpr std::array<unsigned, 3> params{1, 2, 3};

  for (unsigned i = 0; i < 3; i++) {
    if (test_rank == MPIWrapper::global_root_rank) {
      broadcast_context_change(change[i], params[i], test_comm);
    } else {
      auto out = receive_context_change(test_comm);
      CHECK(out[0] == change[i]);
      CHECK(out[1] == params[i]);
    }
  }
}
#endif

void MPIWrapper::send_gso(unsigned int full_n, const double *const mu,
                          MPI_Comm comm) MPI_DIST_MAY_THROW {
  CPUCOUNT(709);
  THROW_OR_OPTIMISE(mu == nullptr, std::invalid_argument,
                    "mu must not be null");
  THROW_OR_OPTIMISE(full_n == 0, std::invalid_argument, "full_n must not be 0");

  // MPI doesn't let you (easily) probe a broadcast to work out how
  // much data is needed on receipt. To fix that, we just forward
  // the size as one broadcast and then the actual data as another.
  broadcast_context_change(ContextChange::LOAD_GSO, full_n, comm);
  MPI_Bcast(const_cast<double *>(mu), static_cast<int>(full_n), MPI_DOUBLE,
            MPIWrapper::global_root_rank, comm);
  TRACK(ContextChange::LOAD_GSO, sizeof(double) * full_n);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("send_gso_tests", 2) {
  // Throws on either null input or a full_n of zero.
  std::array<double, 5> arr{5.0, 9.6, 4.5, 1.2, 9};
  SUBCASE("Throws on invalid inputs") {
    CHECK_THROWS_WITH_AS(MPIWrapper::send_gso(0, arr.data(), test_comm),
                         "full_n must not be 0", std::invalid_argument);
    CHECK_THROWS_WITH_AS(MPIWrapper::send_gso(5, nullptr, test_comm),
                         "mu must not be null", std::invalid_argument);
  }

  SUBCASE("Works otherwise") {
    if (test_rank == MPIWrapper::global_root_rank) {
      MPIWrapper::send_gso(5, arr.data(), test_comm);
    } else {
      std::array<double, 5> in_arr;
      auto status = MPIWrapper::receive_command(test_comm);
      CHECK(status[0] == ContextChange::LOAD_GSO);
      CHECK(status[1] == 5);
      MPI_Bcast(in_arr.data(), 5, MPI_DOUBLE, MPIWrapper::global_root_rank,
                test_comm);
      CHECK(in_arr == arr);
    }
  }
}
#endif

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("receive_gso_no_header", 2) {
  std::array<double, 5> arr{5.0, 9.6, 4.5, 1.2, 9};
  if (test_rank == MPIWrapper::global_root_rank) {
    MPI_Bcast(arr.data(), arr.size(), MPI_DOUBLE, MPIWrapper::global_root_rank,
              test_comm);
  } else {
    std::vector<double> gso;
    MPIWrapper::receive_gso_no_header(5, gso, test_comm);
    CHECK(memcmp(arr.data(), gso.data(), sizeof(double) * 5) == 0);
  }
}
#endif

void MPIWrapper::receive_gso_no_header(const unsigned size,
                                       std::vector<double> &gso,
                                       MPI_Comm comm) noexcept {
  CPUCOUNT(709);
  gso.resize(size);
  MPI_Bcast(gso.data(), static_cast<int>(size), MPI_DOUBLE,
            MPIWrapper::global_root_rank, comm);
}

void MPIWrapper::receive_gso(std::vector<double> &gso, MPI_Comm comm) noexcept {
  const auto status = MPIWrapper::receive_command(comm);
  assert(status[0] == ContextChange::LOAD_GSO);
  MPIWrapper::receive_gso_no_header(status[1], gso, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("send_and_recv_gso", 2) {
  constexpr static auto size = 5;

  std::array<double, size> mu{1, 965.3, 3.14159, 2.7189};
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::send_gso(size, mu.data(), test_comm);
  } else {
    std::vector<double> out;
    MPIWrapper::receive_gso(out, test_comm);
    REQUIRE(out.size() == size);
    for (unsigned i = 0; i < size; i++) {
      CHECK(out[i] == mu[i]);
    }
  }
}
#endif

void MPIWrapper::send_status(const unsigned status, MPI_Comm comm) noexcept {
  broadcast_context_change(ContextChange::CHANGE_STATUS, status, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("send_status", 2) {
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::send_status(0, test_comm);
  } else {
    const auto recv = receive_context_change(test_comm);
    CHECK(recv[0] == ContextChange::CHANGE_STATUS);
    CHECK(recv[1] == 0);
  }
}
#endif

void MPIWrapper::broadcast_params(const SieverParams &params,
                                  const unsigned long seed,
                                  MPI_Comm comm) noexcept {
  CPUCOUNT(710);
  // Note: it's really very tempting to make this function serialise
  // the entire params object using an MPI datatype. This becomes a little bit
  // painful, because we also need to pack the simhash string. Packing directly
  // is slightly easier in this case, and it also saves some additional MPI
  // calls.
  // Work out how much space we need.
  int params_size;
  auto data_type = Layouts::get_param_layout();
  MPI_Pack_size(1, data_type, comm, &params_size);

  int seed_size;
  MPI_Pack_size(1, MPI_UINT64_T, comm, &seed_size);

  int simhash_dirs_size;
  MPI_Pack_size(params.simhash_codes_basedir.size(), MPI_CHAR, comm,
                &simhash_dirs_size);

  // Now send the size to all of the rest of the nodes.
  auto size = params_size + seed_size + simhash_dirs_size;

  // We send the string size separately to the full size, so that the
  // reconstruction on the other side is safe.

  std::array<int, 2> sizes{size, simhash_dirs_size};
  MPI_Bcast(sizes.data(), 2, MPI_INT, MPIWrapper::global_root_rank, comm);

  // Now we have to pack and send.
  int position{};
  std::vector<char> buffer(size);
  MPI_Pack(&params, 1, data_type, buffer.data(), buffer.size(), &position,
           comm);

  // Convert to uint64_t: unsigned long is allowed to vary in
  // representation across machines.
  uint64_t seed_as_uint64 = static_cast<uint64_t>(seed);

  MPI_Pack(&seed_as_uint64, 1,
           Layouts::get_data_type<decltype(seed_as_uint64)>(), buffer.data(),
           buffer.size(), &position, comm);

  MPI_Pack(params.simhash_codes_basedir.data(), simhash_dirs_size, MPI_CHAR,
           buffer.data(), buffer.size(), &position, comm);

  MPI_Bcast(buffer.data(), size, MPI_PACKED, MPIWrapper::global_root_rank,
            comm);

  TRACK(ContextChange::BROADCAST_PARAMS, sizeof(sizes) + size);
  MPI_Type_free(&data_type);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("broadcast_params", 2) {
  SieverParams start{};
  // G6K doesn't set this by default.
  start.reserved_n = 100;
  // randomness for the rest is fine.
  start.bdgl_improvement_db_ratio = 1.9;
  start.bgj1_improvement_db_ratio = 1.9;
  start.otf_lift = false;
  start.saturation_ratio = 9;
  // threads is never changed.
  start.threads = 0;
  // simhash_codes_basedir is changes
  start.simhash_codes_basedir = "test";

  unsigned int long seed = 5;

  if (test_rank == MPIWrapper::global_root_rank) {
    // Change it to prove it doesn't get overwritten.
    start.threads = 45;
    MPIWrapper::broadcast_params(start, seed, test_comm);

  } else {
    SieverParams out{};
    unsigned int long recv_seed;
    auto type = Layouts::get_param_layout();
    MPI_Bcast(reinterpret_cast<char *>(&out), 1, type,
              MPIWrapper::global_root_rank, test_comm);
    CHECK(out.reserved_n == start.reserved_n);
    CHECK(out.bdgl_improvement_db_ratio == start.bdgl_improvement_db_ratio);
    CHECK(out.bgj1_improvement_db_ratio == start.bgj1_improvement_db_ratio);
    CHECK(out.otf_lift == start.otf_lift);
    CHECK(out.saturation_ratio == start.saturation_ratio);
    CHECK(out.threads != start.threads);
    CHECK(out.simhash_codes_basedir != start.simhash_codes_basedir);
    MPI_Type_free(&type);

    uint64_t size;
    MPI_Bcast(&size, 1, MPI_UINT64_T, MPIWrapper::global_root_rank, test_comm);
    CHECK(size == start.simhash_codes_basedir.size());

    std::vector<char> in(size);
    MPI_Bcast(in.data(), size, MPI_CHAR, MPIWrapper::global_root_rank,
              test_comm);
    std::string res(in.data(), size);
    CHECK(res == start.simhash_codes_basedir);

    MPI_Bcast(&recv_seed, 1, MPI_UINT64_T, MPIWrapper::global_root_rank,
              test_comm);
    CHECK(recv_seed == seed);
  }
}
#endif

void MPIWrapper::receive_params(SieverParams &params, uint64_t &seed,
                                MPI_Comm comm) noexcept {
  CPUCOUNT(711);

  // Receive the sizes from the root node.
  // These are packed as (full size, size of simhash_codes_basedir_string).
  std::array<int, 2> sizes;
  MPI_Bcast(sizes.data(), 2, MPI_INT, MPIWrapper::global_root_rank, comm);

  // We allocate just enough space and then unpack.
  std::vector<char> buffer(sizes[0]);

  MPI_Bcast(buffer.data(), buffer.size(), MPI_PACKED,
            MPIWrapper::global_root_rank, comm);

  // Now unpack.
  auto params_type = Layouts::get_param_layout();
  int position{};
  MPI_Unpack(buffer.data(), buffer.size(), &position, &params, 1, params_type,
             comm);
  MPI_Type_free(&params_type);

  MPI_Unpack(buffer.data(), buffer.size(), &position, &seed, 1, MPI_UINT64_T,
             comm);

  // And finally, the string.
  std::vector<char> tmp_str(sizes[1]);
  MPI_Unpack(buffer.data(), buffer.size(), &position, tmp_str.data(), sizes[1],
             MPI_CHAR, comm);

  params.simhash_codes_basedir = std::string(tmp_str.data(), sizes[1]);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("receive_params", 2) {
  SieverParams start{};
  // G6K doesn't set this by default.
  start.reserved_n = 100;
  // randomness for the rest is fine.
  start.bdgl_improvement_db_ratio = 1.9;
  start.bgj1_improvement_db_ratio = 1.9;
  start.otf_lift = false;
  start.saturation_ratio = 9;
  // threads is never changed.
  start.threads = 0;
  // simhash_codes_basedir changes on the remote ranks.
  start.simhash_codes_basedir = "test";

  unsigned int long seed = 5;

  if (test_rank == MPIWrapper::global_root_rank) {
    // Change it to prove it doesn't get overwritten.
    start.threads = 45;
    auto type = Layouts::get_param_layout();
    MPI_Bcast(&start, 1, type, MPIWrapper::global_root_rank, test_comm);
    MPI_Type_free(&type);
    uint64_t str_size = start.simhash_codes_basedir.size();
    MPI_Bcast(&str_size, 1, MPI_UINT64_T, MPIWrapper::global_root_rank,
              test_comm);
    MPI_Bcast(start.simhash_codes_basedir.data(), static_cast<int>(str_size),
              MPI_CHAR, MPIWrapper::global_root_rank, test_comm);
    MPI_Bcast(&seed, 1, MPI_UINT64_T, MPIWrapper::global_root_rank, test_comm);
  } else {
    SieverParams out{};
    unsigned int long recv_seed;
    MPIWrapper::receive_params(out, recv_seed, test_comm);
    CHECK(out.reserved_n == start.reserved_n);
    CHECK(out.bdgl_improvement_db_ratio == start.bdgl_improvement_db_ratio);
    CHECK(out.bgj1_improvement_db_ratio == start.bgj1_improvement_db_ratio);
    CHECK(out.otf_lift == start.otf_lift);
    CHECK(out.saturation_ratio == start.saturation_ratio);
    CHECK(out.threads != start.threads);
    CHECK(out.simhash_codes_basedir == start.simhash_codes_basedir);
    CHECK(recv_seed == seed);
  }
}
#endif

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("send_and_recv_params", 2) {
  // This function is essentially a replication of send_params and
  // receive_params, but with no known working parts.
  SieverParams start{};
  // G6K doesn't set this by default.
  start.reserved_n = 100;
  // randomness for the rest is fine.
  start.bdgl_improvement_db_ratio = 1.9;
  start.bgj1_improvement_db_ratio = 1.9;
  start.otf_lift = false;
  start.saturation_ratio = 9;
  // threads is never changed.
  start.threads = 0;
  // simhash_codes_basedir changes
  start.simhash_codes_basedir = "test";
  unsigned int long seed = 5;

  if (test_rank == MPIWrapper::global_root_rank) {
    // Change it to prove it doesn't get overwritten.
    start.threads = 45;
    MPIWrapper::broadcast_params(start, seed, test_comm);
  } else {
    SieverParams out{};
    unsigned int long recv_seed;
    MPIWrapper::receive_params(out, recv_seed, test_comm);
    CHECK(out.reserved_n == start.reserved_n);
    CHECK(out.bdgl_improvement_db_ratio == start.bdgl_improvement_db_ratio);
    CHECK(out.bgj1_improvement_db_ratio == start.bgj1_improvement_db_ratio);
    CHECK(out.otf_lift == start.otf_lift);
    CHECK(out.saturation_ratio == start.saturation_ratio);
    CHECK(out.threads != start.threads);
    CHECK(out.simhash_codes_basedir == start.simhash_codes_basedir);
    CHECK(recv_seed == seed);
  }
}
#endif

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("receive_initial_setup", 2) {
  SieverParams start{};
  // G6K doesn't set this by default.
  start.reserved_n = 100;
  // randomness for the rest is fine.
  start.bdgl_improvement_db_ratio = 1.9;
  start.bgj1_improvement_db_ratio = 1.9;
  start.otf_lift = false;
  start.saturation_ratio = 9;
  // threads is never changed.
  start.threads = 0;
  // simhash_codes_basedir changes
  start.simhash_codes_basedir = "test";
  std::array<double, 5> arr{5.0, 9.6, 4.5, 1.2, 9};

  unsigned int long seed = 5;

  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::broadcast_params(start, seed, test_comm);
    MPIWrapper::send_gso(5, arr.data(), test_comm);
  } else {
    SieverParams out;
    std::vector<double> gso;
    unsigned int long recv_seed;
    MPIWrapper::receive_initial_setup(out, gso, recv_seed, test_comm);
    CHECK(gso.size() == 5);
    CHECK(out.reserved_n == start.reserved_n);
    CHECK(out.bdgl_improvement_db_ratio == start.bdgl_improvement_db_ratio);
    CHECK(out.bgj1_improvement_db_ratio == start.bgj1_improvement_db_ratio);
    CHECK(out.otf_lift == start.otf_lift);
    CHECK(out.saturation_ratio == start.saturation_ratio);
    CHECK(out.threads != start.threads);
    CHECK(out.simhash_codes_basedir == start.simhash_codes_basedir);
    CHECK(memcmp(gso.data(), arr.data(), sizeof(double) * 5) == 0);
    CHECK(recv_seed == seed);
  }
}
#endif

void MPIWrapper::receive_initial_setup(SieverParams &p,
                                       std::vector<double> &gso, uint64_t &seed,
                                       MPI_Comm comm) noexcept {
  CPUCOUNT(712);
  MPIWrapper::receive_params(p, seed, comm);
  MPIWrapper::receive_gso(gso, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("broadcast_context_change", 2) {
  std::array<ContextChange, 3> change{ContextChange::EL, ContextChange::ER,
                                      ContextChange::SL};
  std::array<unsigned, 3> params{1, 2, 3};

  for (unsigned i = 0; i < 3; i++) {
    if (test_rank == MPIWrapper::global_root_rank) {
      broadcast_context_change(change[i], params[i], test_comm);
    } else {
      std::array<unsigned, 2> arr;
      MPI_Bcast(arr.data(), 2, MPI_UNSIGNED, MPIWrapper::global_root_rank,
                test_comm);
      CHECK(arr[0] == change[i]);
      CHECK(arr[1] == params[i]);
    }
  }
}
#endif

void MPIWrapper::broadcast_el(const unsigned lp, MPI_Comm comm) noexcept {
  broadcast_context_change<true>(ContextChange::EL, lp, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("broadcast_el", 2) {

  const unsigned lp = 1;
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::broadcast_el(lp, test_comm);
  } else {
    std::array<unsigned, 2> received;
    MPI_Bcast(received.data(), 2, MPI_UNSIGNED, MPIWrapper::global_root_rank,
              test_comm);
    CHECK(received[0] == ContextChange::EL);
    CHECK(received[1] == lp);
  }
}
#endif

void MPIWrapper::broadcast_er(const unsigned rp, MPI_Comm comm) noexcept {
  broadcast_context_change<true>(ContextChange::ER, rp, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("broadcast_er", 2) {
  const auto rp = 1;
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::broadcast_er(rp, test_comm);
  } else {
    std::array<unsigned, 2> packed;
    MPI_Bcast(packed.data(), 2, MPI_UNSIGNED, MPIWrapper::global_root_rank,
              test_comm);
    CHECK(packed[0] == ContextChange::ER);
    CHECK(packed[1] == rp);
  }
}
#endif

void MPIWrapper::broadcast_sl(const unsigned lp, const bool down_sieve,
                              MPI_Comm comm) noexcept {
  broadcast_context_change<true>((down_sieve) ? ContextChange::SL_REDIST
                                              : ContextChange::SL_NO_REDIST,
                                 lp, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("broadcast_sl", 2) {
  const auto lp = 1;
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::broadcast_sl(lp, test_comm);
  } else {
    std::array<unsigned, 2> received;
    MPI_Bcast(received.data(), 2, MPI_UNSIGNED, MPIWrapper::global_root_rank,
              test_comm);
    CHECK(received[0] == ContextChange::SL);
    CHECK(received[1] == lp);
  }
}
#endif

static void find_shortest_best_lifts(void *in_vec, void *in_out_vec, int *len,
                                     MPI_Datatype *) {

  // This function works out which of the best lifts are shortest between
  // `in_vec` and `in_out_vec`. As the name suggests, in_out_vec is the location
  // where the outputs need to be written.

  // This function works as follows. We expect that both `in_vec` and
  // `in_out_vec` are packed LiftEntry objects.
  // This assumption is a little bit tricky: MPI reduction routines are allowed
  // to accept any sort of type. For the sake of this code, we need to assert
  // that type is of a very particular form, since this will allow us to decode
  // the array correctly. We need this for all such calls, as the best lifts
  // size depends on the size of the sieving instance (and not on
  // MAX_SIEVING_DIM). In other words, we cannot easily statically bound this
  // function.
  //
  // To circumvent this, we use the pack each lift `l` as <full_n, l.len, l.x>.
  // Including full_n in each seems necessary for MPI to work consistently.

  // First of all, we need to treat in_vec and in_out_vec as char*, which is
  // what they're defined as.
  char *in_lifts = reinterpret_cast<char *>(in_vec);

  // N.B This is just a slight name truncation, but it's still an in_out_vec.
  char *out_lifts = reinterpret_cast<char *>(in_out_vec);
  const unsigned length = static_cast<unsigned>(*len);

  // As a sanity check, we check that the first element of both in_lifts and
  // in_out_lifts are the same.
  unsigned in_lifts_full_n, out_lifts_full_n;
  memcpy(&in_lifts_full_n, in_lifts, sizeof(in_lifts_full_n));
  memcpy(&out_lifts_full_n, out_lifts, sizeof(out_lifts_full_n));
  assert(in_lifts_full_n == out_lifts_full_n);

  // Now we can work out how many bytes each vector will contain.
  const auto size_of_vector =
      in_lifts_full_n * sizeof(ZT) + sizeof(in_lifts_full_n) + sizeof(FT);

  unsigned curr{};
  unsigned in_lifts_n, out_lifts_n;
  FT in_lifts_len, out_lifts_len;

  for (unsigned i = 0; i < length; i++, curr += size_of_vector) {
    memcpy(&in_lifts_n, in_lifts + curr, sizeof(in_lifts_n));
    memcpy(&out_lifts_n, out_lifts + curr, sizeof(out_lifts_n));
    memcpy(&in_lifts_len, in_lifts + curr + sizeof(in_lifts_n),
           sizeof(in_lifts_len));
    memcpy(&out_lifts_len, out_lifts + curr + sizeof(in_lifts_n),
           sizeof(out_lifts_len));

    // Check the sizes are as we expect.
    assert(in_lifts_n == in_lifts_full_n);
    assert(out_lifts_n == out_lifts_full_n);

    // If the lift is valid, copy it over.
    if (in_lifts_len != 0.0 && in_lifts_len < out_lifts_len) {
      memcpy(out_lifts + curr + sizeof(in_lifts_n) + sizeof(FT),
             in_lifts + curr + sizeof(in_lifts_n) + sizeof(FT),
             sizeof(ZT) * in_lifts_full_n);
    }
  }
}

MPI_Op MPIWrapper::get_best_lifts_op() noexcept {
  MPI_Op best_lifts_op;
  // N.B the 1 here means the operation is commutative.
  // This has the advantage of meaning that MPI can potentially re-order how it
  // does this work from a data movement perspective, which can lead to certain
  // speedups.
  MPI_Op_create(&find_shortest_best_lifts, 1, &best_lifts_op);
  return best_lifts_op;
}

void MPIWrapper::reduce_best_lifts_to_root(
    const std::vector<LiftEntry> &lifts_in, std::vector<LiftEntry> &lifts_out,
    const unsigned full_n, const int my_rank, MPI_Op op,
    MPI_Comm comm) noexcept {

  int rank;
  MPI_Comm_rank(comm, &rank);

  CPUCOUNT(713);
  // We know that each party has an array of lifts. Each party has exactly
  // the same number of lifts. We combine these using a custom MPI op.

  // Warning: it appears that sometimes we have various writes "in flight" here
  // that can cause the best_lifts vector to get a bit screwed up. To fix that,
  // we issue a full barrier here. This does increase the cost somewhat, but
  // this is less expensive that the entire sieve crashing.
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // Note: to do this we have to unpack the vectors. This is because MPI
  // doesn't really work well with the double indirection. This shouldn't be
  // too expensive though.
  const unsigned size = lifts_in.size();

  // This function works as follows:
  // 1) We pack each best lift into a buffer, which we serialise. For MPI
  // reasons (see below) we pack each lift `l` as <full_n, l.len, l.x>.
  //    If l.x is empty, we place full_n many 0s into the buffer.
  // 2) We use a custom MPI operation to calculate the best lifts. These are
  // collected only at the root. 3) The root then unpacks the best lifts across
  // the cluster.
  //
  // Q: Why do we pack full_n for each lift?
  // A: Because for some reason, using a custom data type to store `full_n`
  // seems to fail on multiple MPI implementations.
  //    Moreover, as the reduction operation does not have access to the
  //    underlying communicator, we cannot use MPI_Pack for this task.
  //
  // This does require the use of a custom MPI datatype.

  static_assert(std::is_same_v<decltype(LiftEntry::len), FT>,
                "Error: LiftEntry::len is no longer FT");
  static_assert(std::is_same_v<decltype(LiftEntry::x)::value_type, ZT>,
                "Error: LiftEntry::x no longer contains ZT");

  // Each vector contains full_n * sizeof(ZT) bytes (for the coefficients) +
  // sizeof(FT) (for the length) + sizeof(full_n) (for the full_n).
  const auto size_of_vector = full_n * sizeof(ZT) + sizeof(FT) + sizeof(full_n);

  // In total the packed_lifts buffer contains size * size_of_vector bytes.
  std::vector<char> packed_lifts(size_of_vector * size);

  unsigned curr{};
  for (unsigned i = 0; i < size; i++, curr += size_of_vector) {
    memcpy(packed_lifts.data() + curr, &full_n, sizeof(full_n));
    memcpy(packed_lifts.data() + curr + sizeof(full_n), &lifts_in[i].len,
           sizeof(FT));

    // N.B The size here has to be influenced by lifts_in[i].x.size(), as if the
    // lift is empty then lifts_in[i].x will be empty.
    memcpy(packed_lifts.data() + curr + sizeof(full_n) + sizeof(FT),
           lifts_in[i].x.data(), sizeof(ZT) * lifts_in[i].x.size());
  }

  // And now we make the MPI datatype.
  constexpr int count = 3;
  const int block_lengths[]{1, 1, static_cast<int>(full_n)};
  constexpr MPI_Aint offsets[]{0, sizeof(full_n), sizeof(FT) + sizeof(full_n)};
  const MPI_Datatype types[]{Layouts::get_data_type<decltype(full_n)>(),
                             Layouts::get_data_type<FT>(),
                             Layouts::get_data_type<ZT>()};

  MPI_Datatype best_lifts_type;
  MPI_Type_create_struct(count, block_lengths, offsets, types,
                         &best_lifts_type);

  // Now it's time for something a bit confusing:
  // we need to make sure that the "extent" of the datatype is the same as what
  // we've just serialised. The "extent" of a datatype is loosely defined as "If
  // I had this type as a struct, how large would that type be?". In other
  // words, the extent of the type includes padding. You can think of this
  // adjustment as us making sure that MPI reads the data as we've written it
  // i.e without any extra padding. This is necessary because the MPI type
  // creation engine follows the same rules as a compiler would, but we do not
  // (i.e we pack without caring about any struct padding that would be
  // introduced if we were serialising a real type).
  MPI_Aint lb, extent;
  MPI_Type_get_extent(best_lifts_type, &lb, &extent);
  // N.B This cast is fine, because MPI_Aint is (sometimes) a long int,
  // rather than an unsigned long int.
  // N.B N.B The "remove_const_t" is to remove a GCC warning about return type
  // of casts. The warning is harmless, just tedious.
  const auto tmp_extent =
      static_cast<std::remove_const_t<decltype(size_of_vector)>>(extent);
  if (tmp_extent != size_of_vector) {
    auto told = best_lifts_type;
    MPI_Type_create_resized(told, 0, size_of_vector, &best_lifts_type);
    MPI_Type_free(&told);
  }

  MPI_Type_commit(&best_lifts_type);

  TRACK(ContextChange::BEST_LIFTS, size * size_of_vector);

  // And now we just run the routine as is.
  if (my_rank == MPIWrapper::global_root_rank) {
    // If we're the root then we need to run the operation and then copy
    // the results back over into a format that G6K expects.
    // N.B We use packed_best_lifts as a temporary buffer.
    MPI_Reduce(MPI_IN_PLACE, packed_lifts.data(), static_cast<int>(size),
               best_lifts_type, op, MPIWrapper::global_root_rank, comm);

    lifts_out.resize(size);

    // Now we just have to unpack all of the received entries.
    // Everything is contiguous already, so unpacking is the same as
    // packing, just in reverse.

    curr = 0;
    std::remove_const_t<decltype(size)> tmp_size;

    for (unsigned i = 0; i < size; i++, curr += size_of_vector) {
      // We first unpack the full_n element. This should be full_n.
      memcpy(&tmp_size, packed_lifts.data() + curr, sizeof(tmp_size));
      assert(tmp_size == full_n);

      // Now we unpack the length directly into the lifts.
      memcpy(&lifts_out[i].len, packed_lifts.data() + curr + sizeof(tmp_size),
             sizeof(FT));

      // And finally we copy over the lift if it was valid.
      if (lifts_out[i].len != 0.0) {
        lifts_out[i].x.resize(full_n);
        memcpy(lifts_out[i].x.data(),
               packed_lifts.data() + curr + sizeof(tmp_size) + sizeof(FT),
               sizeof(ZT) * full_n);
      }
    }

  } else {
    // By contrast here we don't overwrite the memory, as we don't get the
    // results.
    MPI_Reduce(packed_lifts.data(), nullptr, static_cast<int>(size),
               best_lifts_type, op, MPIWrapper::global_root_rank, comm);
  }

  MPI_Type_free(&best_lifts_type);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("reduce_best_lifts_to_root", 2) {
  // This function works as follows:
  // 1. The root rank generates 2n random best lifts, each of size 64 with
  // random lengths.
  // 2. The root rank sends n of them to the second rank.
  // 3. Both parties reduce.
  // 4. We test at the end that the lengths are as expected.
  constexpr auto n = 120;
  constexpr auto size = 64;

  MPI_Op op{MPIWrapper::get_best_lifts_op()};

  std::vector<LiftEntry> random_best_lifts_one, random_best_lifts_two;
  if (test_rank == MPIWrapper::global_root_rank) {
    random_best_lifts_one.resize(n);
    random_best_lifts_two.resize(n);
    for (unsigned i = 0; i < n; i++) {
      random_best_lifts_one[i].x.resize(size);
      random_best_lifts_two[i].x.resize(size);
      for (unsigned j = 0; j < size; j++) {
        random_best_lifts_one[i].x[j] = static_cast<ZT>(rand());
        random_best_lifts_two[i].x[j] = static_cast<ZT>(rand());
      }
      random_best_lifts_one[i].len = rand();
      random_best_lifts_two[i].len = rand();
    }

    for (unsigned i = 0; i < n; i++) {
      MPI_Send(random_best_lifts_one[i].x.data(), size, MPI_UINT16_T, 1, 0,
               test_comm);
      MPI_Send(&random_best_lifts_one[i].len, 1, MPI_DOUBLE, 1, 0, test_comm);
    }

    std::vector<LiftEntry> result(n);
    MPIWrapper::reduce_best_lifts_to_root(random_best_lifts_two, result, size,
                                          test_rank, op, test_comm);

    for (unsigned i = 0; i < n; i++) {
      const auto *const curr =
          (random_best_lifts_one[i].len < random_best_lifts_two[i].len)
              ? &random_best_lifts_one[i]
              : &random_best_lifts_two[i];
      CHECK(result[i].x == curr->x);
      CHECK(result[i].len == curr->len);
    }

  } else {
    random_best_lifts_one.resize(n);
    for (unsigned i = 0; i < n; i++) {
      random_best_lifts_one[i].x.resize(size);
      MPI_Recv(random_best_lifts_one[i].x.data(), size, MPI_UINT16_T, 0, 0,
               test_comm, MPI_STATUS_IGNORE);
      MPI_Recv(&random_best_lifts_one[i].len, 1, MPI_DOUBLE, 0, 0, test_comm,
               MPI_STATUS_IGNORE);
    }

    MPIWrapper::reduce_best_lifts_to_root(random_best_lifts_one,
                                          random_best_lifts_two, size,
                                          test_rank, op, test_comm);
  }
  MPI_Op_free(&op);
}
#endif

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("get_full_database_size", 2) {

  const int size_1 = 5;
  const int size_2 = 10;
  if (test_rank == MPIWrapper::global_root_rank) {
    std::vector<int> sizes;
    MPIWrapper::get_full_database_size(sizes, size_1, test_comm);
    REQUIRE(sizes.size() == 2);
    CHECK(sizes[0] == size_1);
    CHECK(sizes[1] == size_2);
  } else {
    MPIWrapper::send_database_size_to_root(size_2, test_comm);
  }
}
#endif

void MPIWrapper::send_database_size_to_root(const int total,
                                            MPI_Comm comm) noexcept {
  CPUCOUNT(714);
  // This function is more straightforward: we just need to call gather.
  MPI_Gather(&total, 1, MPI_INT, nullptr, 1, MPI_INT,
             MPIWrapper::global_root_rank, comm);
}

void MPIWrapper::get_full_database_size(std::vector<int> &sizes, const int size,
                                        MPI_Comm comm) noexcept {
  CPUCOUNT(715);
  // First of all we need to gather the number of ranks in the communicator
  // and the rank of this rank. We could pass this in, but it seems cleaner to
  // do it this way.
  int nr_ranks;
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nr_ranks);

  // To make it so we can use inplace, we'll dump our share into the
  // output buffer after resizing.
  sizes.resize(static_cast<unsigned>(nr_ranks));
  sizes[static_cast<unsigned>(my_rank)] = size;

  // N.B only the root rank gets the results, so a gather is fine here.
  MPI_Gather(MPI_IN_PLACE, 1, MPI_INT, sizes.data(), 1, MPI_INT,
             MPIWrapper::global_root_rank, comm);
}

void MPIWrapper::get_database(std::vector<Entry> &db, const unsigned n,
                              MPI_Comm comm) noexcept {
  CPUCOUNT(716);
  // Firstly we need to find out how many ranks there are.
  int nr_ranks;
  MPI_Comm_size(comm, &nr_ranks);

  // Now we'll get the number of vectors there are in the database globally.
  std::vector<int> sizes;

  // Check the cast is safe.
  assert(db.size() <= std::numeric_limits<int>::max());

  // Note: this only needs to be called by the root rank.
  MPIWrapper::get_full_database_size(sizes, static_cast<int>(db.size()), comm);
  TRACK_NO_COUNT(ContextChange::GET_DB, db.size() * sizeof(int));

  // This is the place where things get tricky.
  // Essentially, we use MPI_Gatherv to allow us to collect a varying amount of
  // data from each rank. This is primarily to deal with the case where
  // different ranks potentially have different amounts of data in their
  // databases. In this situation, we need to use the data type that is
  // non-contiguous for a vector in [0, n). This is non-contiguous because we
  // simply send the vectors "as is", and so the structures aren't fully
  // serialised. MPI can handle this, but we have to introduce a custom type for
  // this.
  auto type = Layouts::get_entry_vector_layout(n);

  // We now need to know at which offset from `db` we'll start writing the
  // entries for each rank. Since we'll treat this similarly to other MPI
  // operations (i.e the writes are in order) this corresponds to first writing
  // rank 0's data, then rank 1's data, and so on. This means we need to compute
  // the partial sums of each of the entries in `sizes`. N.B Normally we'd do
  // this lazily, but MPI needs the whole array to be precomputed.
  const auto nr_sizes = sizes.size();
  std::vector<int> offsets(nr_sizes);
  int total{};
  for (unsigned i = 0; i < nr_sizes; i++) {
    offsets[i] = total;
    total += sizes[i];
  }

  // Now we'll resize the database. This is so we can have enough space
  // for each receive.
  // We'll need this for later.
  assert(db.size() <= std::numeric_limits<int>::max());
  const int old_size = static_cast<int>(db.size());

  // N.B this cast must be safe, because total is an int.
  db.resize(static_cast<unsigned>(total));

  // Now we can set everything up into one call.
  // We know that the displacements are in `offsets`. Similarly,
  // we know how much to send and receive from each rank: that's kept in
  // the `sizes` array. This means that all we need to do is line everything up.
  // To make this high-level, we'll use MPI_Gatherv.
  // Note: the MPI_IN_PLACE here means that the data for this rank is already in
  // the database in the right place. It turns out this is true: this is part of
  // the motivation for setting the global_root_rank to 0, since this means we
  // don't need to move around the existing database entries here during the
  // resizing.
  // N.B the "old_size" and the first invocation of "type" are ignored here due
  // to the use of MPI_IN_PLACE.
  MPI_Gatherv(MPI_IN_PLACE, old_size, type, db.data(), sizes.data(),
              offsets.data(), type, MPIWrapper::global_root_rank, comm);

  TRACK(ContextChange::GET_DB,
        Layouts::get_entry_size(n) *
                std::accumulate(sizes.cbegin(), sizes.cend(), uint64_t(0)) -
            db.size());

  // Now everything is in the `db`, so we're all done.
  // We just need to free the type to prevent leaks.
  MPI_Type_free(&type);
}

void MPIWrapper::send_database_to_root(const std::vector<Entry> &db,
                                       const unsigned n,
                                       MPI_Comm comm) noexcept {
  CPUCOUNT(717);
  // This one is more straightforward.  We need to first participate in the
  // size gathering portion of the protocol.
  assert(db.size() <= std::numeric_limits<int>::max());
  const int db_size = static_cast<int>(db.size());
  MPIWrapper::send_database_size_to_root(db_size, comm);
  // And then we just do a gather to let the root get everything.
  // We of course need the MPI type.
  auto type = Layouts::get_entry_vector_layout(n);
  MPI_Gatherv(db.data(), db_size, type, nullptr, nullptr, nullptr, type,
              MPIWrapper::global_root_rank, comm);
  MPI_Type_free(&type);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("build_database_at_root", 2) {
  constexpr auto size = 128;
  const unsigned n = 70;

  std::vector<Entry> db(size);
  std::vector<Entry> db_2(size);
  for (unsigned i = 0; i < size; i++) {
    auto &entry = db[i];
    auto &entry_2 = db_2[i];
    std::iota(entry.x.begin(), entry.x.begin() + n, 0);
    std::iota(entry_2.x.begin(), entry_2.x.begin() + n, 5);
  }

  if (test_rank == MPIWrapper::global_root_rank) {
    auto write_db = db;
    MPIWrapper::get_database(write_db, n, test_comm);
    REQUIRE(write_db.size() == 2 * size);
    for (unsigned i = 0; i < size; i++) {
      CHECK(write_db[i].x == db[i].x);
      CHECK(write_db[i + size].x == db_2[i].x);
    }
  } else {
    MPIWrapper::send_database_to_root(db_2, n, test_comm);
  }
}
#endif

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("split_database", 2) {
  // We'll first generate a random
  // database.
  constexpr auto size = 128;
  const unsigned n = 70;

  std::vector<Entry> db(size);
  unsigned val = 0;
  for (auto &entry : db) {
    std::iota(entry.x.begin(), entry.x.begin() + n, ++val);
  }

  // Now we'll assume there's a random amount of memory per rank.
  // Our general strategy is to express this in terms of the number of vectors
  // each rank can hold.
  std::vector<uint64_t> memory_per_rank;
  SUBCASE("Divides evenly") { memory_per_rank = {64, 64}; }
  SUBCASE("Divides unevenly") { memory_per_rank = {90, 38}; }
  SUBCASE("More than in the database") { memory_per_rank = {150, 175}; }

  // Back up so we can check for correctness later.
  const auto copy = db;

  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::split_database(db, n, memory_per_rank, test_comm);
  } else {
    MPIWrapper::receive_split_database(db, n, test_comm);
  }

  uint64_t other_size;
  auto my_size = static_cast<uint64_t>(db.size());
  if (test_rank == MPIWrapper::global_root_rank) {
    MPI_Send(&my_size, 1, MPI_UINT64_T, 1, 0, test_comm);
    MPI_Recv(&other_size, 1, MPI_UINT64_T, 1, 0, test_comm, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(&other_size, 1, MPI_UINT64_T, 0, 0, test_comm, MPI_STATUS_IGNORE);
    MPI_Send(&my_size, 1, MPI_UINT64_T, 0, 0, test_comm);
  }

  // Check the sizes add up
  CHECK(other_size + my_size == size);

  // Check the elements were actually distributed.
  unsigned start_pos = (test_rank == MPIWrapper::global_root_rank)
                           ? 0
                           : static_cast<unsigned>(other_size);

  for (unsigned i = 0; i < my_size; i++) {
    CHECK(copy[i + start_pos].x == db[i].x);
  }
}
#endif

static std::vector<int>
compute_split(const uint64_t db_size,
              const std::vector<uint64_t> &memory_per_rank) noexcept {

  // Make sure the caller has passed in a valid set.
  assert(memory_per_rank.size() != 0);
  // This function computes how to split the database into a reasonably
  // equitable share between the attached ranks.
  // We firstly work out what fraction of the total memory each rank has.
  const auto total_memory = std::accumulate(
      memory_per_rank.cbegin(), memory_per_rank.cend(), uint64_t(0));
  const auto nr_ranks = memory_per_rank.size();
  std::vector<double> frac_per_rank(nr_ranks);
  for (unsigned i = 0; i < nr_ranks; i++) {
    frac_per_rank[i] = static_cast<double>(memory_per_rank[i]) /
                       static_cast<double>(total_memory);
  }

  // Now we have a series of rounded fractions.
  // We'll allocate the database such that each rank gets a number
  // of vectors proportional to their storage.
  // As per normal, MPI only likes ints.
  int allocated = 0;
  std::vector<int> vectors_per_rank(nr_ranks);
  for (unsigned i = 0; i < nr_ranks; i++) {
    // N.B the cast of db_size loses precision.
    const auto per_rank =
        std::floor(static_cast<double>(db_size) * frac_per_rank[i]);

    assert(per_rank <= std::numeric_limits<int>::max());
    vectors_per_rank[i] = static_cast<int>(per_rank);
    allocated += vectors_per_rank[i];
  }

  // This seems to be sensible if we're rounding down.
  assert(static_cast<uint64_t>(allocated) <= db_size);
  // Now we need to patch up the result.
  // Because we round down in the previous loop, we may not have covered the
  // entire database. In that situation, we add the extra unallocated vectors to
  // the rank that has the most memory.
  // NOTE: this is deterministic. This means that the same rank will always
  // get the extra vectors. We could be clever and schedule this if it matters
  // (it shouldn't, because the sieving dimension here should be small).
  if (db_size - static_cast<unsigned>(allocated) > 0) {
    const auto iter =
        std::max_element(memory_per_rank.cbegin(), memory_per_rank.cend());
    assert(iter != std::cend(memory_per_rank));
    const auto index =
        static_cast<unsigned>(std::distance(memory_per_rank.cbegin(), iter));
    const uint64_t new_for_top_rank =
        static_cast<uint64_t>(vectors_per_rank[index]) +
        (db_size - static_cast<uint64_t>(allocated));
    assert(new_for_top_rank <= std::numeric_limits<int>::max());
    vectors_per_rank[index] = static_cast<int>(new_for_top_rank);
  }

  return vectors_per_rank;
}

void MPIWrapper::split_database(std::vector<Entry> &db, const unsigned n,
                                const std::vector<uint64_t> &memory_per_rank,
                                MPI_Comm comm) noexcept {

  CPUCOUNT(718);
  // We'll work out what the exact split is elsewhere.
  const uint64_t db_size = db.size();
  assert(!memory_per_rank.empty());
  const auto vectors_per_rank = compute_split(db_size, memory_per_rank);
  const auto nr_ranks = memory_per_rank.size();

  // Now we'll actually split the database. This splitting is exactly how you'd
  // expect it to be: the root gets the first `even_share *
  // memory_per_rank[0]` ranks, then the next rank gets the next `even_share *
  // memory_per_rank[1]` vectors, and so on. We can do this succinctly using
  // MPI_Scatterv.
  auto type = Layouts::get_entry_vector_layout(n);
  std::vector<int> offsets(nr_ranks);
  int total{};

  // Produce the partial sum offsets.
  for (unsigned i = 0; i < nr_ranks; i++) {
    offsets[i] = total;
    total += vectors_per_rank[i];
  }

  // MPI collective communications don't let us (easily) probe the communication
  // as with (say) point-to-point communications. To fix that, we broadcast the
  // size to each rank first.
  MPI_Scatter(vectors_per_rank.data(), 1, MPI_INT, MPI_IN_PLACE, 1, MPI_INT,
              MPIWrapper::global_root_rank, comm);

  // Now we'll forward the database as we've decided.
  MPI_Scatterv(db.data(), vectors_per_rank.data(), offsets.data(), type,
               MPI_IN_PLACE, static_cast<int>(db_size), type,
               MPIWrapper::global_root_rank, comm);

  // With this done, there's two tasks left to do:
  // 1. We should shrink the database to reflect the new size.
  // This only works if the root rank is 0.
  static_assert(MPIWrapper::global_root_rank == 0,
                "Error: split_database expects global_root_rank == 0");
  const auto size_for_root =
      static_cast<unsigned>(vectors_per_rank[MPIWrapper::global_root_rank]);
  TRACK(ContextChange::SPLIT_DB,
        Layouts::get_entry_size(n) * (db.size() - size_for_root));

  db.resize(size_for_root);
  // 2. We should free the old MPI type.
  MPI_Type_free(&type);
}

void MPIWrapper::receive_split_database(std::vector<Entry> &db,
                                        const unsigned n,
                                        MPI_Comm comm) noexcept {
  CPUCOUNT(719);
  // The code here is far simpler than in the sending routine. We first receive
  // the number of vectors we will have, and then we resize the database to hold
  // them, and finally actually read them into the database.
  auto type = Layouts::get_entry_vector_layout(n);

  // We first need to gather how much data we are going to receive.
  // This predates the database message.
  int size;
  MPI_Scatter(nullptr, 1, MPI_INT, &size, 1, MPI_INT,
              MPIWrapper::global_root_rank, comm);
  assert(size > 0);
  db.resize(static_cast<unsigned>(size));
  // N.B the plethora of nullptr here is because the receiver doesn't care about
  // these at all.
  MPI_Scatterv(nullptr, nullptr, nullptr, type, db.data(), size, type,
               MPIWrapper::global_root_rank, comm);

  MPI_Type_free(&type);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("split_database_ordered", 2) {
  // This test makes sure the database can be split in an ordered fashion.
  constexpr auto size = 128;
  const unsigned n = 70;

  std::vector<CompressedEntry> cdb(size);
  std::vector<Entry> db(size);
  unsigned val = 0;
  for (auto &entry : db) {
    // For now, this is reasonable.
    cdb[val].i = val;
    std::iota(entry.x.begin(), entry.x.begin() + n, ++val);
  }

  // Now we'll assume there's a random amount of memory per rank.
  // Our general strategy is to express this in terms of the number of vectors
  // each rank can hold.
  std::vector<uint64_t> memory_per_rank;
  SUBCASE("Divides evenly") { memory_per_rank = {64, 64}; }
  SUBCASE("Divides unevenly") { memory_per_rank = {90, 38}; }
  SUBCASE("More than in the database") { memory_per_rank = {150, 175}; }

  // Copy these for later.
  auto cdb_copy = cdb;
  auto db_copy = db;

  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::split_database_ordered(cdb, db, n, memory_per_rank, test_comm);
  } else {
    MPIWrapper::receive_database_ordered(cdb, db, n, test_comm);
  }

  REQUIRE(db.size() == cdb.size());

  uint64_t other_size;
  auto my_size = static_cast<uint64_t>(db.size());
  if (test_rank == MPIWrapper::global_root_rank) {
    MPI_Send(&my_size, 1, MPI_UINT64_T, 1, 0, test_comm);
    MPI_Recv(&other_size, 1, MPI_UINT64_T, 1, 0, test_comm, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(&other_size, 1, MPI_UINT64_T, 0, 0, test_comm, MPI_STATUS_IGNORE);
    MPI_Send(&my_size, 1, MPI_UINT64_T, 0, 0, test_comm);
  }

  // Check the sizes add up
  CHECK(other_size + my_size == size);

  // Check the elements were actually distributed.
  const unsigned start_pos = (test_rank == MPIWrapper::global_root_rank)
                                 ? 0
                                 : static_cast<unsigned>(other_size);

  for (unsigned i = 0; i < my_size; i++) {
    const Entry &dbe = db_copy[cdb_copy[i + start_pos].i];
    const Entry &dbm = db[cdb[i].i];
    CHECK(dbe.x == dbm.x);
  }
}
#endif

void MPIWrapper::split_database_ordered(
    std::vector<CompressedEntry> &cdb, std::vector<Entry> &db, const unsigned n,
    const std::vector<uint64_t> &memory_per_rank, MPI_Comm comm) noexcept {
  CPUCOUNT(720);
  const uint64_t db_size = db.size();
  // Seems reasonable.
  assert(cdb.size() == db_size);

  // We'll work out what an equitable split would be.
  assert(!memory_per_rank.empty());
  const auto vectors_per_rank = compute_split(db_size, memory_per_rank);

  // The next part is trickier. Essentially, whilst we have a reasonably
  // straightforward way of sending the _cdb_ entries, this is not the same as
  // sending the _db_ entries in a straightforward way.
  // The primary reason for this is because the db entries are not sorted,
  // and so there's no reasonable, natural ordering for this.
  // For the sake of this function, we expect that the overhead here is small:
  // this is because we expect the initial splitting here to occur for small
  // database sizes, and according to the literature
  // (https://arxiv.org/pdf/1809.10778.pdf), messages smaller than 10^8 bytes
  // perform equivalently for any non-contiguous mechanism. We hence use the
  // high-level API to deal with these non-contiguous sends, since at worst it
  // performs the same (and at best it performs faster).

  // This still leaves us with the problem of _how_ to actually pack the
  // discontiguous vectors.
  // Thankfully, MPI_Type_struct is sufficiently general that doing this packing
  // is somewhat easy.
  // The high-level approach is this:
  // 1. We create a new MPI type for each sender.
  // 2. We use the indices from the compressed database to work out the offsets
  //    at each point.

  const auto nr_ranks = memory_per_rank.size();
  assert(nr_ranks != 0);
  // We don't send anything to ourselves, so we
  // just focus on the other ranks.
  static_assert(
      MPIWrapper::global_root_rank == 0,
      "Error: this function requires MPIWrapper::global_root_rank == 0");

  // The -1 here is because, as mentioned, we skip the root rank.
  std::vector<MPI_Datatype> types(nr_ranks - 1);
  std::vector<MPI_Request> requests(nr_ranks - 1);
  auto curr = cdb.cbegin() + vectors_per_rank[MPIWrapper::global_root_rank];

  for (unsigned i = 0; i < nr_ranks - 1; i++) {
    // You should view this as saying: we shall send all of the entries in `db`
    // indexed by the `cdb` between `curr` and `curr + vectors_per_rank`.
    types[i] = Layouts::get_entry_layout_non_contiguous(
        curr, curr + vectors_per_rank[i + 1], n);
    curr += vectors_per_rank[i + 1];
  }

  // We've covered them all.
  assert(curr == cdb.cend());
  // Send them to the outside world.
  for (unsigned i = 0; i < nr_ranks - 1; i++) {
    MPI_Isend(db.data(), 1, types[i], static_cast<int>(i + 1),
              static_cast<int>(ContextChange::ORDERED_DB_SPLIT), comm,
              &requests[i]);
  }

  // Now we need to free our types.
  for (auto &type : types) {
    MPI_Type_free(&type);
  }

  // And then we'll wait for the results.
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
              MPI_STATUSES_IGNORE);

  // And finally we'll resize the database to make sure everything lines up.
  // Note that the _only_ thing guaranteed here is that the indices are correct!
  const auto new_db_size =
      static_cast<uint64_t>(vectors_per_rank[MPIWrapper::global_root_rank]);

  TRACK(ContextChange::ORDERED_DB_SPLIT,
        Layouts::get_entry_size(n) * (db.size() - new_db_size));

  for (unsigned i = 0; i < new_db_size; i++) {
    db[i] = db[cdb[i].i];
    cdb[i].i = i;
  }

  // And resize.
  db.resize(new_db_size);
  cdb.resize(new_db_size);
}

void MPIWrapper::receive_database_ordered(std::vector<CompressedEntry> &cdb,
                                          std::vector<Entry> &db,
                                          const unsigned n,
                                          MPI_Comm comm) noexcept {
  CPUCOUNT(720);
  // Here we just use a blocking recv to make sure we return properly.
  // But first we'll check how much data we expect.
  MPI_Status status;
  MPI_Probe(MPIWrapper::global_root_rank,
            static_cast<int>(ContextChange::ORDERED_DB_SPLIT), comm, &status);
  auto type = Layouts::get_entry_vector_layout(n);
  int amount;
  MPI_Get_count(&status, type, &amount);
  // This must be true.
  assert(amount > 0);

  // Now we'll resize and read.
  cdb.resize(static_cast<unsigned>(amount));
  db.resize(static_cast<unsigned>(amount));

  // This might be confusing if this is the first time you've seen this:
  // the sending data type (i.e the type built in send_database_ordered) is not
  // the same type we're using here. It turns out this is OK in MPI: the
  // important thing is that MPI needs to be able to map the values that are
  // being sent to the values that are being received. Since we're sending and
  // receiving entries in both directions, then this is OK.
  MPI_Recv(db.data(), amount, type, MPIWrapper::global_root_rank,
           static_cast<int>(ContextChange::ORDERED_DB_SPLIT), comm,
           MPI_STATUS_IGNORE);

  // And now make sure you can actually access them through the
  // cdb. This does not imply that the rest of the entries are initialised.
  for (unsigned i = 0; i < static_cast<unsigned>(amount); i++) {
    cdb[i].i = i;
  }

  MPI_Type_free(&type);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("db_size", 2) {
  const size_t first = 10;
  const size_t second = 50;

  size_t out;
  if (test_rank == MPIWrapper::global_root_rank) {
    out = MPIWrapper::db_size(first, test_comm);
  } else {
    out = MPIWrapper::db_size(second, test_comm);
  }

  CHECK(out == first + second);
}
#endif

size_t MPIWrapper::db_size(const size_t size, MPI_Comm comm) noexcept {
  CPUCOUNT(721);
  uint64_t out = static_cast<uint64_t>(size);
  MPI_Allreduce(MPI_IN_PLACE, &out, 1, MPI_UINT64_T, MPI_SUM, comm);
#ifdef MPI_TRACK_BANDWIDTH
  {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    TRACK(ContextChange::DB_SIZE, sizeof(out) * comm_size);
  }
#endif

  return static_cast<size_t>(out);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("global_saturation", 2) {
  const size_t sats[2]{10, 50};
  size_t out = MPIWrapper::global_saturation(sats[test_rank], test_comm);
  CHECK(out == sats[0] + sats[1]);
}
#endif

size_t MPIWrapper::global_saturation(const size_t sat, MPI_Comm comm) noexcept {
  CPUCOUNT(722);
  uint64_t out = static_cast<uint64_t>(sat);
  MPI_Allreduce(MPI_IN_PLACE, &out, 1, MPI_UINT64_T, MPI_SUM, comm);
#ifdef MPI_TRACK_BANDWIDTH
  {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    TRACK(ContextChange::GET_GLOBAL_SATURATION, comm_size * sizeof(out));
  }
#endif
  return static_cast<size_t>(out);
}

static size_t
reserve_or_grow_db_impl(const size_t N,
                        const std::vector<uint64_t> &memory_per_rank,
                        const ContextChange tag, MPI_Comm comm) noexcept {
  // In practice this is statically true because only functions in this TU
  // invoke this, so the compiler should be smart enough to optimise this away.
  assert(tag == ContextChange::GROW_SMALL || tag == ContextChange::GROW_LARGE ||
         tag == ContextChange::RESERVE);

  // Similarly to with split_db, we work out what a "fair" split
  // is for the change.
  assert(!memory_per_rank.empty());
  const auto division = compute_split(N, memory_per_rank);

  // To make the event loop easier to program on receiving ranks, we do two
  // collective operations here: a broadcast to indicate that we are in a grow /
  // shrink situation and then the values themselves as a scatter. This is
  // probably the most performant and simple way to do this, see:
  // https://stackoverflow.com/questions/74460245/is-there-a-way-to-do-disjoint-reads-of-mpi-bcast
  // The 0 below is just a placeholder for "don't care"
  broadcast_context_change(tag, 0, comm);
  MPI_Scatter(division.data(), 1, MPI_INT, MPI_IN_PLACE, 1, MPI_INT,
              MPIWrapper::global_root_rank, comm);

  TRACK(tag, (division.size() - 1) * sizeof(int));

  // Return the size for the root rank's db after this call.
  return static_cast<size_t>(division[MPIWrapper::global_root_rank]);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("shrink_or_grow_db_impl", 2) {
  // This test case just checks the shrink_or_grow_db function. This exists
  // solely to make testing easier.
  std::vector<uint64_t> memory_per_rank;
  // Initialised to stop the compiler complaining: the
  // warning is wrong, but the compiler can't
  // figure that out.
  ContextChange t{ContextChange::GROW_SMALL};
  // N.B these overlapping subcases are because doctest only executes a single
  // subcase at once: so we need to do it for each high-level strategy. We could
  // move to table-based testing if this is too annoying.
  SUBCASE("Divides evenly") {
    memory_per_rank = {64, 64};
    SUBCASE("GrowSmall") { t = ContextChange::GROW_SMALL; }
    SUBCASE("GrowLarge") { t = ContextChange::GROW_LARGE; }
    SUBCASE("Shrink") { t = ContextChange::SHRINK; }
  }
  SUBCASE("Divides unevenly") {
    memory_per_rank = {90, 38};
    SUBCASE("GrowSmall") { t = ContextChange::GROW_SMALL; }
    SUBCASE("GrowLarge") { t = ContextChange::GROW_LARGE; }
    SUBCASE("Shrink") { t = ContextChange::SHRINK; }
  }
  SUBCASE("More than in the database") {
    memory_per_rank = {150, 175};
    SUBCASE("GrowSmall") { t = ContextChange::GROW_SMALL; }
    SUBCASE("GrowLarge") { t = ContextChange::GROW_LARGE; }
    SUBCASE("Shrink") { t = ContextChange::SHRINK; }
  }

  static constexpr auto new_size = 128;
  if (test_rank == MPIWrapper::global_root_rank) {
    auto out = shrink_or_grow_db_impl(new_size, memory_per_rank, t, test_comm);
    // This is just for testing: we read the size received by the other rank.
    int other;
    MPI_Recv(&other, 1, MPI_UINT64_T, 1, 0, test_comm, MPI_STATUS_IGNORE);
    CHECK(static_cast<size_t>(other) + out == new_size);
  } else {
    // First the root broadcasts two values. We can use
    // receive_context_change to receive this. The first contains the header
    // and the second holds nothing useful.
    const auto vals = receive_context_change(test_comm);
    CHECK(vals[0] == t);

    // Then we get the actual size from a scatter.
    int in;
    MPI_Scatter(nullptr, 1, MPI_INT, &in, 1, MPI_INT,
                MPIWrapper::global_root_rank, test_comm);
    // We then send the value to the other rank so it can check the sizes add
    // up.
    MPI_Send(&in, 1, MPI_INT, MPIWrapper::global_root_rank, 0, test_comm);
  }
}
#endif

size_t MPIWrapper::grow_db(const size_t N, const unsigned large,
                           const std::vector<uint64_t> &memory_per_rank,
                           MPI_Comm comm) noexcept {
  CPUCOUNT(724);
  return reserve_or_grow_db_impl(
      N, memory_per_rank,
      (large ? ContextChange::GROW_LARGE : ContextChange::GROW_SMALL), comm);
}

size_t MPIWrapper::reserve_db(const size_t N,
                              const std::vector<uint64_t> &memory_per_rank,
                              MPI_Comm comm) noexcept {
  CPUCOUNT(724);
  return reserve_or_grow_db_impl(N, memory_per_rank, ContextChange::RESERVE,
                                 comm);
}

FT MPIWrapper::get_norm(const size_t N, const FT min, const FT max,
                        const std::vector<CompressedEntry> &cdb,
                        const ContextChange tag, MPI_Comm comm) noexcept {

  // This function approximately finds the norm of the `N`-th element in the
  // global database from many sorted lists. This is done by applying a binary
  // search over multiple sorted lists. This algorithm appears to be a variant
  // of a much older algorithm by Iyer, Ricard and Varman (introduced in
  // "Percentile Finding Algorithm for Multiple Sorted Runs"), but there appears
  // to be enough differences to make this not an exact copy. Thus, this
  // deserves some sort of explanation.
  //
  // At a high-level, the goal of this algorithm is to find the norm of the N-th
  // shortest vector globally without merging the individual databases.

  // The strategy we take is similar to
  // a distributed priority queue: we iteratively account for the shortest
  // vectors that are spread across the processes, and then recurse. This
  // happens by allowing each process to manage its own list: if process `j`
  // accounts for `n` vectors, it's as if we've removed `n` vectors from the
  // global priority queue.

  // Here's a more concrete description.
  // As setup, we gather the global minimum (g_min) and maximum (g_max) norms in
  // the database. Each process `i` sets up a pair of iterators l_i, u_i that
  // point to the beginning and end of their cdbs respectively. Then,
  // iteratively:
  // 1. The root process broadcasts mid = (g_min + g_max) / 2.0 to all other
  // processes.
  // 2. Each process finds the index of the first element (m_i) in the range
  // (l_i, u_i) that has a norm greater than mid, returning
  //    d_i = m_i - l_i. The root learns d = \sum d_i. The searching is done via
  //    binary search.
  // 3. If d > N, then we replace g_max with mid. Each process sets their u_i
  // to m_i, which shrinks the range. Intuitively, this
  //    is the same as discarding elements that are going to be larger than the
  //    N-th entry.
  // 4. If d <= N, then we replace g_min with mid. Each process sets their l_i
  // to m_i. At this stage, we also replace N with N - d.
  //    This might seem counter-intuitive, but the idea is the same as in step
  //    3: we know that the elements we've searched so far cannot contain the
  //    N-th entry. You can view this as recursively solving the same problem,
  //    but over smaller lists.
  // 5. If N == 0 at this stage, then we have found our n-th entry. Note that
  // each process
  //    may have a non-zero range at this point (i.e l_i may not be u_i). This
  //    is fine: each attached process will shrink their database accordingly.
  // As far as the algorithm seems to go, these extra vectors either all have
  // the same norm, or it is such a small difference that the deviation does not
  // matter. Indeed, experimentally the algorithm appears to always return a
  // length that leads to a global database size that is exactly correct.

  // A natural question is: why not just do this over lengths? That is, why not
  // just do a binary search over [g_min, g_max] until we have a norm that means
  // we retain the best `N` elements? The reason is complicated, but essentially
  // we can end up with a situation where the binary search does not terminate:
  // we cannot keep reducing g_mid indefinitely (we implemented this initially).
  // This is partly a compression issue: cdb elements store their norms as
  // 32-bit floats, but this isn't accurate enough in some settings. We also
  // have this issue in this code, but because we also track the iterator
  // distance we can detect this issue straightforwardly.

  // The algorithm's runtime is upper bounded polynomially in the dimension of
  // the lattice. Notice that we can analyse the algorithm as if we were doing a
  // binary search over one large list: this requires O(log n) iterations in the
  // worst case. As the list held by each process `p` contains at most n
  // entries, we have a process-centric runtime of O(|p| * log n). As `n` is
  // approximately 2^(0.210d), we get O(|p| * 0.210d) as the running time,
  // ignoring the size of each list. If we do not ignore the size of the lists,
  // then we can see that each inner binary search costs at most O(log n) steps.
  // This means in total we have ~ |p| * O(log n) * O(log n) steps, which
  // translates to O(|p| * d^2) steps in total. In other words, this function
  // should be cheap enough to run even on large instances.

  int rank;
  MPI_Comm_rank(comm, &rank);

  // We first gather the minimum and maximum lengths.
  // This is fairly straightforward to do with
  // a single MPI call: we negate the minimum length to allow us to use the
  // pre-defined MPI_MAX.
  FT g_min, g_max;
  {
    std::array<FT, 2> lengths{-min, max};

    // MPI requires that MPI_IN_PLACE is only used on the root process for
    // MPI_Reduce: in all other situations, we need to pass in the actual data.
    auto send_buf =
        (rank == MPIWrapper::global_root_rank) ? MPI_IN_PLACE : lengths.data();
    MPI_Reduce(send_buf, lengths.data(), 2, Layouts::get_data_type<FT>(),
               MPI_MAX, MPIWrapper::global_root_rank, comm);

#ifdef MPI_TRACK_BANDWIDTH
    if (rank == MPIWrapper::global_root_rank) {
      int comm_size;
      MPI_Comm_size(comm, &comm_size);
      TRACK_NO_COUNT(tag, comm_size * sizeof(lengths));
    }
#endif
    g_min = -lengths[0];
    g_max = lengths[1];
  }

  // Each process has three iterators that we use: a begin, an end, and a
  // middle.
  auto start = cdb.cbegin(), end = cdb.cend();
  // This starts at cdb.cbegin() to make life slightly easier when writing
  // the main loop.
  auto mid = cdb.cbegin();

  // We use 0.0 as a stop condition: if a process receives a norm of 0.0, then
  // we consider the loop finished. Here's a long comment block on why 0.0: Use
  // of 0:
  // ----------------
  // Using 0 as a stopping value is reasonable because:
  // (1) We never expect a length that is that much shorter than the Gaussian
  // Heuristic (all of the vectors are normalised in their lengths), and (2)
  // Both the C standards and the IEEE 754 standards (i.e the widely-used
  // floating point standard) imply that the representation
  //     of zero is essentially an all zero bit pattern. Thus, sending this
  //     allows for a quick comparison elsewhere. In practice, this is a regular
  //     (double) FP comparison.
  // Put differently, 0 is never a legitimate value in a database, and
  // comparison is portable.
  //
  // Why not NaN?:
  // ------------------
  // There's a subtle bug / feature here that will occur if you replace 0.0
  // with NaN (this is what we used originally).
  // Essentially, checks against NaNs are often optimised away when
  // compiling with -Ofast: this is because -Ofast implies -ffast-math, which
  // causes the compiler to assume that NaN (and Inf) never occur. Thus,
  // if we explicitly check for NaN then the check will always return false, and
  // hence the loop will never terminate. Given that G6K uses -Ofast by default,
  // this is a big issue. See, for example, this link:
  // https://godbolt.org/z/5W5nM44z9 (specifically, the first and second views).
  constexpr FT stop_val = 0.0;

  // With this information we now have all the information we need for a binary
  // search. Specifically, we are looking for a target length that allows us to
  // keep the best `N` entries globally.
  FT length;

  // This is used to cache the value from the previous iteration.
  // We use this when deciding how to move the iterators on
  // each remote process: this means we can use a reduce rather than an
  // AllReduce. N.B This starts at 0.0 so that the first iteration is
  // essentially a no-op (this is also why mid starts at the beginning of the
  // cdb).
  FT old_len = 0.0;

  // We'll re-use this lambda in both loops.
  const auto number_under = [&start, &mid, &end, &old_len](const FT len) {
    if (len > old_len) {
      start = mid;
    } else {
      end = mid;
    }

    mid = std::lower_bound(start, end, len,
                           [](const CompressedEntry &ce, FT const &bound) {
                             return ce.len < bound;
                           });
    old_len = len;
    return mid - start;
  };

  // And these two, too.
  const auto bcast_length = [&length, &comm]() {
    MPI_Bcast(&length, 1, Layouts::get_data_type<decltype(length)>(),
              MPIWrapper::global_root_rank, comm);
  };

  // This array has two purposes:
  // 1) It tracks the number of entries we have under the received bound, and
  // 2) It tracks the distance between end and start. If all processes hit 0 on
  // this count, then we are done.
  std::array<uint64_t, 2> under{};

  const auto gather_under = [&comm, rank, &under]() {
    // N.B The optimiser should be clever enough to remove this check, since
    // it's inside an equivalent block (see below).
    auto send_buf =
        (rank == MPIWrapper::global_root_rank) ? MPI_IN_PLACE : under.data();
    MPI_Reduce(send_buf, under.data(), 2, Layouts::get_data_type<uint64_t>(),
               MPI_SUM, MPIWrapper::global_root_rank, comm);
  };

  if (rank == MPIWrapper::global_root_rank) {
    auto remaining = N;
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    while (remaining != 0) {
      // Find the new mid-point.
      length = (g_min + g_max) / 2.0;
      TRACK_NO_COUNT(tag, sizeof(length));
      bcast_length();
      // Work out how many we have under the bound.
      under[0] = number_under(length);
      under[1] = (end - start);

      // And now work out how many that was globally.
      TRACK_NO_COUNT(tag, sizeof(under) * comm_size);
      gather_under();

      // We now have three cases:
      // (a) if under[0] was non-zero, we just continue.
      // (b) if under[0] was zero and under[1] is non-zero, then we continue.
      // (c) if under[0] and under[1] are both zero, then we've hit a sticking
      // point with precision. We just continue.

      // We now adjust the counts etc depending on if it was above or under.
      if (under[0] == 0 && under[1] == 0) {
        break;
      }

      if (under[0] > remaining) {
        g_max = length;
      } else {
        remaining -= under[0];
        g_min = length;
      }
    }

    // And now we stop everyone else.
    FT stop = stop_val;
    TRACK_NO_COUNT(tag, sizeof(stop));
    MPI_Bcast(&stop, 1, Layouts::get_data_type<decltype(stop)>(),
              MPIWrapper::global_root_rank, comm);
  } else {
    while (true) {
      bcast_length();
      if (length == stop_val)
        break;
      under[0] = number_under(length);
      under[1] = (end - start);
      gather_under();
    }
  }

  // The root rank now holds the target length. We broadcast that globally so
  // all ranks have it.
  MPI_Bcast(&length, 1, Layouts::get_data_type<decltype(length)>(),
            MPIWrapper::global_root_rank, comm);
  TRACK_NO_COUNT(tag, sizeof(length));
  return length;
}

size_t MPIWrapper::shrink_db(const size_t N, const FT min, const FT max,
                             const std::vector<CompressedEntry> &cdb,
                             MPI_Comm comm) noexcept {
  CPUCOUNT(725);

  // This function does two things:
  // (1) It broadcasts the "shrink" message to each attached process.
  // (2) It calculates how much each process should shrink their database by.
  // Broadly speaking, this is carried out in the get_norm function.

  int rank;
  MPI_Comm_rank(comm, &rank);
  // This will be entered first by the root rank, so we just issue the SHRINK
  // message in that case.
  if (rank == MPIWrapper::global_root_rank) {
    // Note: the N != 0 value is to avoid executing the algorithm if we're just
    // throwing everything away. This can happen sometimes.
    // N.B We track the message here, but in get_norm everything is accumulated
    // into this message.
    broadcast_context_change<true>(ContextChange::SHRINK, (N != 0), comm);
    // We return to the caller if we're just throwing everything away.
    if (N == 0)
      return 0;
  }

  // Gather the norm of the Nth element globally.
  const auto length = get_norm(N, min, max, cdb, ContextChange::SHRINK, comm);

  // Work out how many elements we have that are short.
  return std::lower_bound(cdb.cbegin(), cdb.cend(), length,
                          [](const CompressedEntry &ce, const FT &bound) {
                            return ce.len < bound;
                          }) -
         cdb.cbegin();
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("sort_db", 2) {
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::sort_db(test_comm);
  } else {
    const auto cc = receive_context_change(test_comm);
    CHECK(cc[0] == ContextChange::SORT);
    // The second value doesn't matter here.
  }
}
#endif

void MPIWrapper::sort_db(MPI_Comm comm) noexcept {
  broadcast_context_change(ContextChange::SORT, 0, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("receive_command", 2) {
  SUBCASE("single param") {
    constexpr std::array<ContextChange, 5> flags{
        ContextChange::EL, ContextChange::ER, ContextChange::SL,
        ContextChange::SORT, ContextChange::CHANGE_STATUS};
    constexpr unsigned params[]{1, 2, 3, 4, 5};
    for (unsigned i = 0; i < flags.size(); i++) {
      if (test_rank == MPIWrapper::global_root_rank) {
        broadcast_context_change(flags[i], params[i], test_comm);
      } else {
        const auto recv = MPIWrapper::receive_command(test_comm);
        CHECK(recv[0] == flags[i]);
        CHECK(recv[1] == params[i]);
      }
    }
  }

  SUBCASE("multi-param") {
    constexpr std::array<ContextChange, 3> flags{ContextChange::GROW_SMALL,
                                                 ContextChange::GROW_LARGE,
                                                 ContextChange::SHRINK};
    std::array<int, 2> vals{1, 5};
    for (auto flag : flags) {
      if (test_rank == MPIWrapper::global_root_rank) {
        MPI_Bcast(&flag, 1, MPI_UNSIGNED, MPIWrapper::global_root_rank,
                  test_comm);
        MPI_Scatter(vals.data(), 1, MPI_INT, MPI_IN_PLACE, 1, MPI_INT,
                    MPIWrapper::global_root_rank, test_comm);
      } else {
        const auto recv = MPIWrapper::receive_command(test_comm);
        CHECK(recv[0] == flag);
        CHECK(recv[1] == vals[1]);
      }
    }
  }
}
#endif

std::array<unsigned, 2> MPIWrapper::receive_command(MPI_Comm comm) noexcept {
  CPUCOUNT(726);
  const auto out = receive_context_change(comm);

  // Sometimes we also need to read a scattered message. This only happens if
  // it's a large / small grow.
  if (out[0] == ContextChange::GROW_LARGE ||
      out[0] == ContextChange::GROW_SMALL || out[0] == ContextChange::RESERVE) {
    int val;
    MPI_Scatter(nullptr, 1, MPI_INT, &val, 1, MPI_INT,
                MPIWrapper::global_root_rank, comm);
    return {out[0], static_cast<unsigned>(val)};
  }

  return out;
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("gso_update_postprocessing", 2) {
  constexpr unsigned l_ = 49;
  constexpr unsigned r_ = 120;

  constexpr unsigned n = r_ - l_;
  // Size before context change.
  constexpr unsigned old_n = n - 1;

  std::array<long, n * old_n> M;
  std::iota(M.begin(), M.end(), 0);

  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::gso_update_postprocessing(l_, r_, old_n, M.data(), test_comm);
  } else {
    // First thing we receive is a context change.
    const auto recv = receive_context_change(test_comm);
    CHECK(recv[0] == ContextChange::GSO_PP);
    CHECK(recv[1] == r_ - l_);
    std::array<unsigned, 2> l_and_r;
    MPI_Bcast(l_and_r.data(), 2, MPI_UNSIGNED, MPIWrapper::global_root_rank,
              test_comm);
    CHECK(l_and_r[0] == l_);
    CHECK(l_and_r[1] == r_);

    std::array<long, n * old_n> M_new;
    MPI_Bcast(M_new.data(), n * old_n, MPI_LONG, MPIWrapper::global_root_rank,
              test_comm);
    CHECK(M_new == M);
  }
}
#endif

void MPIWrapper::gso_update_postprocessing(const unsigned int l_,
                                           const unsigned int r_,
                                           const unsigned old_n, long const *M,
                                           const bool should_redist,
                                           MPI_Comm comm) noexcept {
  CPUCOUNT(727);
  // First of all we broadcast a message informing all attached ranks about
  // the change. Since they should already hold `n`, we can just send l_ + r_
  // here. N.B this will have the effect of sending this twice, but this helps
  // the other ranks reserve memory for this message.
  broadcast_context_change(ContextChange::GSO_PP, r_ - l_, comm);

  // Then we just send the rest out. The receiving ranks know how much space
  // they need for this, so we can just pack it.
  std::array<unsigned, 3> arr{l_, r_, unsigned(should_redist)};
  MPI_Bcast(arr.data(), 3, MPI_UNSIGNED, MPIWrapper::global_root_rank, comm);
  MPI_Bcast(const_cast<long *>(M), static_cast<int>((r_ - l_) * old_n),
            MPI_LONG, MPIWrapper::global_root_rank, comm);
  TRACK(ContextChange::GSO_PP, sizeof(arr) + sizeof(long) * (r_ - l_) * old_n);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("receive_gso_update_postprocessing", 2) {
  constexpr unsigned l_ = 49;
  constexpr unsigned r_ = 120;

  constexpr unsigned n = r_ - l_;
  // Size before context change.
  constexpr unsigned old_n = n - 1;

  std::array<long, n * old_n> M;
  std::iota(M.begin(), M.end(), 0);
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::gso_update_postprocessing(l_, r_, old_n, M.data(), test_comm);
  } else {
    const auto header = receive_context_change(test_comm);
    CHECK(header[0] == ContextChange::GSO_PP);
    CHECK(header[1] == n);

    std::vector<long> M_new;
    const auto recv =
        MPIWrapper::receive_gso_update_postprocessing(M_new, old_n, test_comm);
    CHECK(recv[0] == l_);
    CHECK(recv[1] == r_);
    CHECK(M_new.size() == n * old_n);
    CHECK(memcmp(M_new.data(), M.data(), sizeof(long) * (n * old_n)) == 0);
  }
}
#endif

std::array<unsigned, 3> MPIWrapper::receive_gso_update_postprocessing(
    std::vector<long> &M, const unsigned old_n, MPI_Comm comm) noexcept {
  CPUCOUNT(728);
  std::array<unsigned, 3> arr;
  MPI_Bcast(arr.data(), 3, MPI_UNSIGNED, MPIWrapper::global_root_rank, comm);
  const auto size = (arr[1] - arr[0]) * old_n;
  M.resize(size);
  MPI_Bcast(M.data(), static_cast<int>(size), MPI_LONG,
            MPIWrapper::global_root_rank, comm);
  return arr;
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("send_stop", 2) {
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::send_stop(test_comm);
  } else {
    const auto status = receive_context_change(test_comm);
    CHECK(status[0] == ContextChange::STOP);
  }
}
#endif

void MPIWrapper::send_stop(MPI_Comm comm) noexcept {
  broadcast_context_change<true>(ContextChange::STOP, 0, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("start_bgj1", 2) {
  constexpr double alpha = 5.0;
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::start_bgj1(alpha, test_comm);
  } else {
    const auto status = receive_context_change(test_comm);
    CHECK(status[0] == ContextChange::BGJ1);
    double alpha_in;
    MPI_Bcast(&alpha_in, 1, MPI_DOUBLE, MPIWrapper::global_root_rank,
              test_comm);
    CHECK(alpha == alpha_in);
  }
}
#endif

void MPIWrapper::start_bgj1(const double alpha, MPI_Comm comm) noexcept {
  broadcast_context_change(ContextChange::BGJ1, 0, comm);
  MPI_Bcast(const_cast<double *>(&alpha), 1, MPI_DOUBLE,
            MPIWrapper::global_root_rank, comm);
  TRACK(ContextChange::BGJ1, sizeof(alpha));
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("receive_alpha", 2) {
  constexpr double alpha = 5.0;
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::start_bgj1(alpha, test_comm);
  } else {
    const auto status = receive_context_change(test_comm);
    CHECK(status[0] == ContextChange::BGJ1);
    const auto alpha_in = MPIWrapper::receive_alpha(test_comm);
    CHECK(alpha == alpha_in);
  }
}
#endif

double MPIWrapper::receive_alpha(MPI_Comm comm) noexcept {
  double alpha;
  MPI_Bcast(&alpha, 1, MPI_DOUBLE, MPIWrapper::global_root_rank, comm);
  return alpha;
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("is_one_g6k", 2) {
  SUBCASE("neither is") {
    MPI_CHECK(0, !MPIWrapper::is_one_g6k(false, test_comm));
    MPI_CHECK(1, !MPIWrapper::is_one_g6k(false, test_comm));
  }

  SUBCASE("one is") {
    MPI_CHECK(0, MPIWrapper::is_one_g6k(true, test_comm));
    MPI_CHECK(1, MPIWrapper::is_one_g6k(false, test_comm));
  }

  SUBCASE("too many are") {
    MPI_CHECK(0, !MPIWrapper::is_one_g6k(true, test_comm));
    MPI_CHECK(1, !MPIWrapper::is_one_g6k(true, test_comm));
  }
}
#endif

bool MPIWrapper::is_one_g6k(const bool is_g6k, MPI_Comm comm) noexcept {
  // We just do a collective sum here.
  int value = is_g6k;
  MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_SUM, comm);
#ifdef MPI_TRACK_BANDWIDTH
  {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    TRACK(ContextChange::IS_ONE_G6K, sizeof(is_g6k) * comm_size);
  }
#endif
  return value == 1;
}

void MPIWrapper::broadcast_initialize_local(const unsigned int ll,
                                            const unsigned int l,
                                            const unsigned int r,
                                            MPI_Comm comm) noexcept {

  broadcast_context_change(ContextChange::IL, ll, comm);
  // Send over the "l and r" portion.
  std::array<unsigned, 2> arr{l, r};
  MPI_Bcast(arr.data(), 2, MPI_UNSIGNED, MPIWrapper::global_root_rank, comm);
  TRACK(ContextChange::IL, sizeof(arr));
}

std::array<unsigned, 2> MPIWrapper::receive_l_and_r(MPI_Comm comm) noexcept {
  std::array<unsigned, 2> arr;
  MPI_Bcast(arr.data(), 2, MPI_UNSIGNED, MPIWrapper::global_root_rank, comm);
  return arr;
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("broadcast_il", 2) {
  constexpr unsigned ll = 1;
  constexpr unsigned l = 2;
  constexpr unsigned r = 3;

  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::broadcast_initialize_local(ll, l, r, test_comm);
  } else {
    const auto status = MPIWrapper::receive_command(test_comm);
    CHECK(status[0] == ContextChange::IL);
    CHECK(status[1] == ll);

    const auto l_and_r = MPIWrapper::receive_l_and_r(test_comm);
    CHECK(l_and_r[0] == l);
    CHECK(l_and_r[1] == r);
  }
}
#endif

void MPIWrapper::build_global_histo(long *const histo, MPI_Comm comm) noexcept {
  CPUCOUNT(731);
  // Just add up the entries.
  // N.B the split here is to allow the root to use MPI_IN_PLACE.
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);
  if (my_rank == MPIWrapper::global_root_rank) {
#ifdef MPI_TRACK_BANDWIDTH
    {
      int comm_size;
      MPI_Comm_size(comm, &comm_size);
      TRACK(ContextChange::HISTO,
            Siever::size_of_histo * comm_size * sizeof(long));
    }
#endif

    MPI_Reduce(MPI_IN_PLACE, histo, Siever::size_of_histo, MPI_LONG, MPI_SUM,
               MPIWrapper::global_root_rank, comm);
  } else {
    MPI_Reduce(histo, nullptr, Siever::size_of_histo, MPI_LONG, MPI_SUM,
               MPIWrapper::global_root_rank, comm);
  }
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("build_global_histo", 2) {
  std::array<long, Siever::size_of_histo> a1;
  std::array<long, Siever::size_of_histo> a2;

  std::iota(a1.begin(), a1.end(), 0);
  std::iota(a2.begin(), a2.end(), 0);
  std::array<long, Siever::size_of_histo> expected;
  for (unsigned i = 0; i < Siever::size_of_histo; i++) {
    expected[i] = a1[i] + a2[i];
  }

  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::build_global_histo(a1.data(), test_comm);
    CHECK(a1 == expected);
  } else {
    MPIWrapper::build_global_histo(a2.data(), test_comm);
  }
}
#endif

void MPIWrapper::broadcast_build_histo(MPI_Comm comm) noexcept {
  broadcast_context_change(ContextChange::HISTO, 0, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("broadcast_build_histo", 2) {
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::broadcast_build_histo(test_comm);
  } else {
    const auto status = MPIWrapper::receive_command(test_comm);
    CHECK(status[0] == ContextChange::HISTO);
  }
}
#endif

void MPIWrapper::broadcast_db_capacity(MPI_Comm comm) noexcept {
  broadcast_context_change(ContextChange::DB_SIZE, 1, comm);
}

void MPIWrapper::broadcast_db_size(MPI_Comm comm) noexcept {
  broadcast_context_change(ContextChange::DB_SIZE, 0, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("broadcast_db_size", 2) {
  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::broadcast_db_size(test_comm);
  } else {
    const auto status = MPIWrapper::receive_command(test_comm);
    CHECK(status[0] == ContextChange::DB_SIZE);
  }
}
#endif

template <typename T>
static double compute_variance(const std::vector<T> &data) noexcept {
  static_assert(
      std::is_arithmetic_v<T>,
      "Error: cannot instantiate compute_variance on a non-numeric type");

  // N.B assumption here that data.size() > 1: this is guarded in
  // gather_gbl_variance.
  const auto size = data.size();
  // N.B another assumption here is that all entries in data are non-negative.
  // This is primarily because for our usage we always expect this to be true.
  const auto mean = std::accumulate(data.cbegin(), data.cend(), T(0)) / size;
  const auto variance_f = [mean, size](T total, const T val) {
    return total + (val - mean) * (val - mean) / (size);
  };

  return std::accumulate(data.cbegin(), data.cend(), 0.0, variance_f);
}

#ifdef MPI_DIST_TEST
TEST_CASE("compute_variance") {
  SUBCASE("single entry") {
    SUBCASE("1") {
      std::vector<int> data(1, 1);
      CHECK(compute_variance(data) == 0.0);
    }

    SUBCASE("50") {
      std::vector<int> data(1, 50);
      CHECK(compute_variance(data) == 0.0);
    }

    SUBCASE("floats") {
      std::vector<float> data(1, 0.0);
      CHECK(compute_variance(data) == 0.0);
    }
  }

  SUBCASE("all same") {
    SUBCASE("floats") {
      std::vector<float> data(50, 1.0);
      CHECK(compute_variance(data) == 0.0);
    }

    SUBCASE("ints") {
      std::vector<int> data(500, 19);
      CHECK(compute_variance(data) == 0.0);
    }
  }

  SUBCASE("different") {
    SUBCASE("ints") {
      // N.B While compute_variance expects non-negative inputs
      // in this case it works out nicely.
      std::vector<int> data = {5, -1, 2};
      CHECK(compute_variance(data) == 6);
    }
  }
}
#endif

template <typename T>
static double gather_gbl_variance(T val, MPI_Comm comm) noexcept {
  static_assert(std::is_arithmetic_v<T>,
                "Error: cannot instantiate gather_gbl_variance with a "
                "non-numeric type");

  // If there's only one of us, then there's no variance at all.
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  if (comm_size == 0) {
    return 0.0;
  }

  // Otherwise, we only collect the variance at the root rank, so we need to
  // know if we are the root rank.
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  if (comm_rank == MPIWrapper::global_root_rank) {
    std::vector<T> entries(comm_size);
    entries[MPIWrapper::global_root_rank] = val;
    // MPI_IN_PLACE says "our value is already in entries".
    MPI_Gather(MPI_IN_PLACE, 1, Layouts::get_data_type<T>(), entries.data(), 1,
               Layouts::get_data_type<T>(), MPIWrapper::global_root_rank, comm);
    TRACK(ContextChange::GET_GLOBAL_VARIANCE, comm_size * sizeof(T));
    return compute_variance(entries);
  } else {
    // Just send the results to the root.
    MPI_Gather(&val, 1, Layouts::get_data_type<T>(), nullptr, 1,
               Layouts::get_data_type<T>(), MPIWrapper::global_root_rank, comm);
    return 0.0;
  }
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("gather_gbl_variance", 3) {
  constexpr std::array<int, 3> var{5, -1, 2};
  MPI_CHECK(0, gather_gbl_variance(var[test_rank], test_comm) == 6);
  MPI_CHECK(1, gather_gbl_variance(var[test_rank], test_comm) == 0.0);
  MPI_CHECK(2, gather_gbl_variance(var[test_rank], test_comm) == 0.0);
}
#endif

double MPIWrapper::gather_gbl_sat_variance(const size_t cur_sat,
                                           MPI_Comm comm) noexcept {
  CPUCOUNT(732);
  // N.B the static_cast here is because size_t doesn't have a corresponding
  // MPI data type.
  return gather_gbl_variance(static_cast<uint64_t>(cur_sat), comm);
}

double MPIWrapper::gather_gbl_ml_variance(const double cur_gbl_ml,
                                          MPI_Comm comm) noexcept {
  CPUCOUNT(733);
  return gather_gbl_variance(cur_gbl_ml, comm);
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("serialise_and_deserialise_stats", 3) {
  SieveStatistics stats{};

  // Just to make it interesting.
  stats.inc_stats_sorting_sieve();

  // Just to check that others also serialise.
  stats.inc_stats_2reds_inner();
  stats.inc_stats_2reds_inner();
  stats.inc_stats_2reds_inner();

  // This block is here just so that tests may pass if
  // statistics gathering is turned off at compile time:
  // if stats gathering is off, then serialise_stats and
  // deserialise_stats_and_add don't do anything other than return.
  // Note: the 0 is because if we can't serialise, then no stats
  // are gathered: so, inc_stats_sorting_sieve above is a no-op.
  // This also assumes that you've enabled stats via ENABLE_EXTENDED_STATS,
  // solely because of the choice of 2reds.
  const auto expected_val =
      (SieveStatistics::can_serialise) ? test_nb_procs : 0;

  if (test_rank == MPIWrapper::global_root_rank) {
    MPIWrapper::deserialise_stats_and_add(stats, test_comm);
    CHECK(stats.get_stats_sorting_sieve() == expected_val);
    CHECK(stats.get_stats_2reds_inner() == 3 * expected_val);
  } else {
    MPIWrapper::serialise_stats(stats, test_comm);
  }
}
#endif

void MPIWrapper::serialise_stats(const SieveStatistics &stats,
                                 MPI_Comm comm) noexcept {
  CPUCOUNT(734);
  if constexpr (!SieveStatistics::can_serialise) {
    return;
  } else {

    // We only send the non-double stats right now.
    std::array<uint64_t, SieveStatistics::number_unsigned_long_stats> longs{};
    stats.serialise_longs(longs);
    MPI_Send(longs.data(),
             static_cast<int>(SieveStatistics::number_unsigned_long_stats),
             MPI_UINT64_T, MPIWrapper::global_root_rank,
             static_cast<int>(ContextChange::STATS_LONG), comm);
  }
}

void MPIWrapper::deserialise_stats_and_add(SieveStatistics &stats,
                                           MPI_Comm comm) noexcept {
  CPUCOUNT(735);
  if constexpr (!SieveStatistics::can_serialise) {
    return;
  } else {

    // We just receive the longs here, similarly to serialise_stats.
    std::array<uint64_t, SieveStatistics::number_unsigned_long_stats> longs{};

    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    TRACK(ContextChange::STATS_LONG, sizeof(longs) * comm_size - 1);

    for (unsigned i = 0; i < comm_size; i++) {
      if (i == MPIWrapper::global_root_rank)
        continue;
      MPI_Recv(longs.data(),
               static_cast<int>(SieveStatistics::number_unsigned_long_stats),
               MPI_UINT64_T, i, static_cast<int>(ContextChange::STATS_LONG),
               comm, MPI_STATUS_IGNORE);
      stats.deserialise_longs_add(longs);
    }
  }
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("get_global_sat_drop", 3) {
  std::array<size_t, 3> sats{1, 1, 1};
  CHECK(MPIWrapper::get_global_sat_drop(sats[test_rank], test_comm) == 2);
}
#endif

void MPIWrapper::reset_stats(MPI_Comm comm) noexcept {
  broadcast_context_change<true>(ContextChange::RESET_STATS, 0, comm);
}

double MPIWrapper::get_min_max_len(double max_len, MPI_Comm comm) noexcept {
#ifdef MPI_TRACK_BANDWIDTH
  {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == MPIWrapper::global_root_rank) {
      int comm_size;
      MPI_Comm_size(comm, &comm_size);
      TRACK(ContextChange::GET_MIN_MAX_LEN, comm_size * sizeof(max_len));
    }
  }
#endif
  MPI_Allreduce(MPI_IN_PLACE, &max_len, 1, MPI_DOUBLE, MPI_MIN, comm);
  return max_len;
}

#ifdef MPI_DIST_TEST
MPI_TEST_CASE("setup_slot_map", 2) {
  std::vector<uint64_t> memory_per_rank{2, 2};
  std::array<int, DB_UID_SPLIT> map;
  MPIWrapper::setup_slot_map(map, memory_per_rank);
  CHECK(map[0] == 0);
  CHECK(map[1 + DB_UID_SPLIT / 2] == 1);
  CHECK(map.back() == 1);
}
#endif

template <typename Container>
static auto split_elems(Container &container,
                        const std::vector<uint64_t> &memory_per_rank) noexcept {
  assert(!memory_per_rank.empty());
  const auto res = compute_split(container.size(), memory_per_rank);
  assert(res.size() == memory_per_rank.size());
  assert(std::accumulate(res.cbegin(), res.cend(), 0) == container.size());

  // We linearly divide elements in order amongst all of the various nodes.
  auto iter = container.begin();
  for (unsigned i = 0; i < res.size(); i++) {
    std::fill(iter, iter + res[i], i);
    iter += res[i];
    assert(iter <= container.cend());
  }

  return res;
}

std::vector<int> MPIWrapper::setup_owner_array(
    std::vector<uint32_t> &vec,
    const std::vector<uint64_t> &memory_per_rank) noexcept {
  return split_elems(vec, memory_per_rank);
}

void MPIWrapper::setup_slot_map(
    std::array<int, DB_UID_SPLIT> &map,
    const std::vector<uint64_t> &memory_per_rank) noexcept {
  split_elems(map, memory_per_rank);
}

void MPIWrapper::receive_database_uids(std::vector<CompressedEntry> &cdb,
                                       std::vector<Entry> &db, const unsigned n,
                                       MPI_Comm comm) noexcept {

  assert(db.empty());
  assert(cdb.empty());

  auto type = Layouts::get_entry_vector_layout(n);
  MPI_Status status;
  MPI_Probe(MPIWrapper::global_root_rank,
            static_cast<int>(ContextChange::DB_SPLIT_UIDS), comm, &status);
  int count;
  MPI_Get_count(&status, type, &count);
  assert(count >= 0);
  db.resize(count);
  cdb.resize(count);

  MPI_Recv(db.data(), count, type, MPIWrapper::global_root_rank,
           static_cast<int>(ContextChange::DB_SPLIT_UIDS), comm,
           MPI_STATUS_IGNORE);

  for (unsigned i = 0; i < static_cast<unsigned>(count); i++) {
    cdb[i].i = i;
  }

  MPI_Type_free(&type);
}

void MPIWrapper::split_database_uids(std::vector<CompressedEntry> &cdb,
                                     std::vector<Entry> &db,
                                     const std::array<int, DB_UID_SPLIT> &map,
                                     const unsigned n, MPI_Comm comm) noexcept {

  // The idea behind this function is to split the database so that particular
  // nodes hold certain portions of the global hash table. We do this by
  // iterating over the db and selecting those entries that belong to a
  // particular rank.
  int comm_size;
  int rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &rank);

  assert(comm_size >= 1);
  assert(rank >= 0);

  // Note: we use CEs here to prevent crashes in cases of large databases.
  std::vector<std::vector<CompressedEntry>> ents(comm_size);
  std::vector<MPI_Request> reqs(comm_size - 1);

  for (const auto &ce : cdb) {
    const auto owner = map[UidHashTable::get_slot(db[ce.i].uid)];
    ents[owner].emplace_back(ce);
  }

  unsigned curr = 0;

  for (unsigned i = 0; i < static_cast<unsigned>(comm_size); i++) {
    if (i == static_cast<unsigned>(rank))
      continue;
    auto &req = reqs[curr];
    auto type = Layouts::get_entry_layout_non_contiguous(ents[i].cbegin(),
                                                         ents[i].cend(), n);
    MPI_Isend(db.data(), 1, type, i,
              static_cast<int>(ContextChange::DB_SPLIT_UIDS), comm, &req);
    MPI_Type_free(&type);
    ++curr;
  }

  MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  TRACK(ContextChange::DB_SPLIT_UIDS,
        Layouts::get_entry_size(n) *
            (db.size() - ents[MPIWrapper::global_root_rank].size()));

  cdb = ents[MPIWrapper::global_root_rank];

  const auto cdb_size = cdb.size();

  std::vector<Entry> my_ents(cdb_size);
  for (unsigned i = 0; i < cdb_size; i++) {
    my_ents[i] = db[cdb[i].i];
  }

  db.resize(cdb_size);

  for (unsigned i = 0; i < cdb_size; i++) {
    db[i] = my_ents[i];
  }

  for (unsigned i = 0; i < cdb_size; i++) {
    cdb[i].i = i;
  }
}

struct DBInfo {
  std::vector<uint64_t> incoming_sizes;
  uint64_t outgoing_total;
};

template <ContextChange tag>
static DBInfo calculate_redist_traffic(std::vector<uint64_t> &outgoing_sizes,
                                       const int rank, MPI_Comm comm) noexcept {

  DBInfo out{};
  const auto comm_size = outgoing_sizes.size();
  out.incoming_sizes.resize(comm_size);
  out.incoming_sizes[rank] = outgoing_sizes[rank];
  MPI_Request req;
  MPI_Ialltoall(outgoing_sizes.data(), 1, MPI_UINT64_T,
                out.incoming_sizes.data(), 1, MPI_UINT64_T, comm, &req);

  TRACK_NO_COUNT(tag, (outgoing_sizes.size() - 1) * sizeof(uint64_t));

  MPI_Wait(&req, MPI_STATUS_IGNORE);
  assert(outgoing_sizes[rank] == 0);
  out.outgoing_total = std::accumulate(outgoing_sizes.cbegin(),
                                       outgoing_sizes.cend(), uint64_t(0));
  return out;
}

class RedistBDGLAlg {
public:
  RedistBDGLAlg(std::vector<CompressedEntry> &cdb_, std::vector<Entry> &db_,
                std::vector<size_t> &sizes_, std::vector<uint32_t> &buckets_,
                const size_t bsize_, const std::vector<uint32_t> &owner_,
                const std::vector<uint32_t> &ours_, const unsigned size,
                const unsigned rank)
      : cdb{cdb_}, db{db_}, sizes{sizes_}, buckets{buckets_}, bsize{bsize_},
        owner{owner_}, ours{ours_},
        outgoing_vectors(size, std::vector<unsigned>()), outgoing_sizes(size),
        bucket_sizes(size * owner.size()), rank(rank) {}

  std::vector<uint64_t> &get_outgoing_sizes() {
    curr_elems.reserve(ours.size());
    for (unsigned i = 0; i < sizes.size(); i++) {
      // The ith bucket is owned by owner[i].
      const auto own = owner[i];

      if (own == rank) {
        curr_elems.emplace_back(sizes[i]);
      } else {
        const auto start = outgoing_vectors[own].size();
        outgoing_vectors[own].resize(start + sizes[i]);
        for (unsigned j = 0; j < sizes[i]; j++) {
          outgoing_vectors[own][start + j] = buckets[bsize * i + j];
        }
      }
    }

    // Now write out those sizes.
    const auto comm_size = outgoing_vectors.size();

    for (unsigned i = 0; i < static_cast<unsigned>(comm_size); i++) {
      if (i == rank) {
        outgoing_sizes[i] = 0;
        continue;
      }
      outgoing_sizes[i] = outgoing_vectors[i].size();
    }

    return outgoing_sizes;
  }

  void recompute_initial_received(unsigned other_rank, uint64_t pos,
                                  std::vector<Entry> &tmp_sends,
                                  std::vector<unsigned> &modified_ces) {
    auto &vec = outgoing_vectors[other_rank];
    auto iter = vec.begin();

    // In this setting, we need to work out which bucket the vectors we've
    // received belong to. We do this by counting down how many we've received
    // from this particular node.
    // For their particular incoming bucket.
    for (unsigned i = 0; i < ours.size(); i++) {
      // Fetch the elements we might be retrieving from.
      auto elem = ours[i];
      auto &remaining = bucket_sizes[other_rank * owner.size() + elem];
      auto start = iter;
      for (; iter < vec.begin() + outgoing_sizes[other_rank] && remaining != 0;
           iter++, remaining--) {
        // Unpack the element by assigning it to the current bucket.
        // This also caps the size of each bucket to prevent overflows.
        if (curr_elems[i] < bsize) {
          buckets[elem * bsize + curr_elems[i]] = *iter;
          ++curr_elems[i];
        }
        used.emplace_back(*iter);
      }

      const auto to_add = std::distance(start, iter);
      if (sizes[elem] + to_add >= bsize) {
        sizes[elem] = bsize;
      } else {
        sizes[elem] += to_add;
      }

      // If we're at the end, then we've finished reading the incoming
      // vectors. We just bail.
      if (iter == vec.end()) {
        return;
      }
    }

    assert(iter == vec.begin() + outgoing_sizes[other_rank]);

    // Otherwise, we send the rest.
    for (unsigned j = 0; iter < vec.end(); iter++, j++) {
      tmp_sends[pos + j] = db[cdb[*iter].i];
      modified_ces.emplace_back(*iter);
    }
  }

  void recompute_extra_received(unsigned other_rank, uint64_t &curr_elem,
                                const uint64_t pos,
                                const std::vector<Entry> &tmp_sends,
                                const std::vector<unsigned> &modified_ces,
                                const size_t outgoing_size,
                                const size_t extra_sends) {

    // If we have more to process, we do the same thing as in the initial
    // receiving loop.
    if (outgoing_vectors[other_rank].size() == outgoing_size) {
      unsigned val{};
      for (unsigned j = 0; j < ours.size(); j++) {
        // Fetch the elements we might be retrieving from.
        auto elem = ours[j];
        auto &remaining = bucket_sizes[other_rank * owner.size() + elem];

        // We always process this many elements.
        size_t can_write = std::min(bsize - sizes[elem], remaining);

        if (sizes[elem] + remaining >= bsize) {
          assert(bsize >= sizes[elem]);
          sizes[elem] = bsize;
        } else {
          sizes[elem] += remaining;
        }

        while (remaining != 0) {
          const auto index = modified_ces[curr_elem++];
          db[cdb[index].i] = tmp_sends[pos + val];

          // Write the position in too, but only if we aren't capped.
          if (can_write != 0) {
            buckets[elem * bsize + curr_elems[j]] = index;
            --can_write;
            assert(curr_elems[j] < bsize);
            ++curr_elems[j];
          }

          used.emplace_back(index);
          --remaining;
          ++val;
        }
        assert(sizes[elem] <= bsize);
      }
    }
  }

  bool should_receive_more(const unsigned i, const size_t outgoing_size) {
    return outgoing_vectors[i].size() == outgoing_size;
  }

  MPI_Datatype get_entry_type(const unsigned n) {
    return Layouts::get_entry_vector_layout(n);
  }

  void swap_initial(const unsigned n, MPI_Comm comm) {
    // In this code, we just pulse the sends into two large buffers.
    // For speed, we serialise these based on their co-efficient representations
    // and unpack.
    const auto comm_size = outgoing_sizes.size();

    // At a high-level, this function works as follows. We allocate two, fixed
    // size buffers (the exact size of these buffers is capped). We then iterate
    // through each set of outgoing insertions and copy back over.
    auto type = Layouts::get_entry_vector_layout(n);

    // N.B We need to store entries directly for this.
    constexpr auto max_size = unsigned(5e7 / sizeof(Entry));
    const auto alloc_size = unsigned(
        std::accumulate(outgoing_sizes.cbegin(), outgoing_sizes.cend(), 0));
    std::vector<Entry> send_buffer(std::min(alloc_size, max_size)),
        recv_buffer(std::min(alloc_size, max_size));

    // These vectors contain the various bits of state for this function.
    // Because we're iterating through a buffer, we need the ability to track
    // which swaps we've already executed. We do this by simply maintaining a
    // counter for each other node and incrementing as appropriate.
    std::vector<int> heads(comm_size, 0);

    // These contain the sending displacements and the number of ZTs we're
    // exchanging per iteration.
    std::vector<int> displs(comm_size, 0);
    std::vector<int> wsizes(comm_size, 0);

    // To prevent trampling, we restrict the maximum amount we can send in each
    // iteration, but only if the buffer size is limited. If the size isn't
    // limited, then we just copy everything in a single iteration.
    const auto max_copy = max_size / comm_size;
    const auto is_capped = send_buffer.size() == max_size;
    assert(max_copy != 0);

    // This lambda deals with the packing / unpacking of vectors. Since the code
    // is mostly the same, we implement both packing and unpacking in one
    // function.
    const auto serial_func = [&wsizes, &displs, &heads, &send_buffer,
                              &recv_buffer, max_copy, comm_size, is_capped,
                              this](const bool pack) {
      // At the beginning of each iteration, we are at the beginning of both the
      // send and receive buffers.
      unsigned buffer_pos{};

      for (unsigned i = 0; i < comm_size; i++) {
        if (i == rank)
          continue;

        // This counts either the number of entries we're sending or receiving.
        // If we're not capped, then we can just copy all of them over:
        // otherwise, we limit ourselves to max_copy.
        const auto to_use =
            (is_capped) ? std::min(max_copy, outgoing_sizes[i] - heads[i])
                        : outgoing_sizes[i];

        // If we're packing, then we need to track that we're sending to_use
        // Entries.
        if (pack) {
          wsizes[i] = to_use;
        }

        // Now handle the (un)packing. This simply tracks the current entries
        // we've already handled. Note that this lambda is somewhat stateful: we
        // do not update heads[i] after sending, only after we've copied over.
        const auto start_pos = heads[i];
        const auto old_pos = buffer_pos;

        for (unsigned j = 0; j < to_use; j++, buffer_pos++) {
          const auto vector_pos = outgoing_vectors[i][j + start_pos];
          auto &ent = db[cdb[vector_pos].i];
          if (pack) {
            send_buffer[buffer_pos] = ent;
          } else {
            ent = recv_buffer[buffer_pos];
          }
        }

        if (pack) {
          displs[i] = old_pos;
        } else {
          heads[i] += to_use;
        }
      }

      assert(buffer_pos <= send_buffer.size());
      assert(buffer_pos <= recv_buffer.size());
    };

    // These are all just neatness helpers.
    const auto pack = [&serial_func]() { serial_func(true); };
    const auto unpack = [&serial_func]() { serial_func(false); };

    const auto done = [&heads, this, comm_size]() {
      for (unsigned i = 0; i < comm_size; i++) {
        if (i != rank && heads[i] != outgoing_sizes[i])
          return false;
      }
      return true;
    };

    do {
      pack();
      // Execute the swaps.
      MPI_Alltoallv(send_buffer.data(), wsizes.data(), displs.data(), type,
                    recv_buffer.data(), wsizes.data(), displs.data(), type,
                    comm);
      unpack();
    } while (!done());
    MPI_Type_free(&type);
  }

  std::vector<Entry> &get_db() { return db; }
  std::vector<CompressedEntry> &get_cdb() { return cdb; }

  void do_extra(MPI_Comm comm) {
    assert(sizes.size() == owner.size());
    std::copy(sizes.cbegin(), sizes.cend(),
              bucket_sizes.begin() + (rank * owner.size()));
    MPI_Allgather(MPI_IN_PLACE, owner.size(),
                  Layouts::get_data_type<uint64_t>(), bucket_sizes.data(),
                  owner.size(), Layouts::get_data_type<uint64_t>(), comm);
    TRACK_NO_COUNT(ContextChange::BDGL_BUCKET, sizeof(uint64_t) * owner.size());
  }

  std::vector<unsigned> get_used() { return used; }

  std::unordered_set<unsigned> get_used_set() {
    return std::unordered_set<unsigned>(used.cbegin(), used.cend());
  }

  template <typename F> void apply_filter(F &&filter_func) {
    filter_func(used);
  }

  void init_for_removal(const unsigned new_size) {
    for (unsigned i = 0; i < ours.size(); i++) {
      const auto elem = ours[i];
      const auto size = sizes[elem];
      const auto start = elem * bsize;
      for (unsigned j = 0; j < size; j++) {
        assert(lookup.find(buckets[j + start]) == lookup.cend());
        lookup[buckets[j + start]] = j + start;
      }
    }
  }

  void postprocess_swap(const unsigned old_val, const unsigned new_val) {
    // This can happen if (for example) the value has been assigned to a bucket
    // that wasn't for us. In such a setting, we never actually use this
    // particular value, and so there's no point updating the index.
    if (lookup.find(old_val) == lookup.cend()) {
      assert([&]() {
        // Check that the owner of old_val was, in fact, not us.
        // This should only happen in some settings.
        const auto iter = std::find(buckets.cbegin(), buckets.cend(), old_val);

        // Not assigned to a bucket.
        if (iter == buckets.cend())
          return true;

        const auto slot = std::distance(buckets.cbegin(), iter);
        const auto owner = slot - (slot % bsize) / sizes.size();
        return owner != rank;
      }());
      return;
    }

    assert(lookup.find(old_val) != lookup.cend());
    const auto index = lookup[old_val];
    buckets[index] = new_val;
    lookup.erase(old_val);
  }

  static constexpr auto TYPE_TAG = ContextChange::BDGL_BUCKET;
  static constexpr auto SEND_TAG = 42;

private:
  std::vector<CompressedEntry> &cdb;
  std::vector<Entry> &db;
  std::vector<size_t> &sizes;
  std::vector<uint32_t> &buckets;
  size_t bsize;

  std::vector<uint32_t> ours;
  const std::vector<uint32_t> &owner;

  std::vector<std::vector<unsigned>> outgoing_vectors;
  std::vector<uint64_t> outgoing_sizes;
  std::vector<uint64_t> bucket_sizes;
  unsigned rank;

  std::vector<unsigned> curr_elems;
  std::vector<unsigned> used;

  std::unordered_map<unsigned, unsigned> lookup;
};

class RedistDBAlg {
public:
  RedistDBAlg(UidHashTable &hash_table_,
              const std::array<int, DB_UID_SPLIT> &slot_map_,
              std::vector<CompressedEntry> &cdb_, std::vector<Entry> &db_,
              std::vector<unsigned> &duplicates_,
              std::vector<unsigned> &incoming_, const unsigned size,
              const unsigned rank_)
      : hash_table{hash_table_}, slot_map{slot_map_}, cdb{cdb_}, db{db_},
        duplicates{duplicates_}, incoming{incoming_},
        outgoing_vectors(size, std::vector<unsigned>()),
        outgoing_sizes(size), rank{rank_} {}

private:
  void recompute_func(const unsigned i) {
    auto &dbe = db[cdb[i].i];
    dbe.uid = hash_table.compute_uid(dbe.x);

    if (slot_map[UidHashTable::get_slot(dbe.uid)] == rank &&
        hash_table.insert_uid(dbe.uid)) {
      incoming.emplace_back(i);
    } else {
      duplicates.emplace_back(i);
    }
  }

public:
  std::vector<uint64_t> &get_outgoing_sizes() {
    // To make things slightly less likely to break, we issue a barrier here.
    // In order to ensure that all writes are actually finished, this needs to
    // be a full barrier. This also means that all of the checks we do after
    // here are fully consistent, so we don't need to worry about error cases.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // Assign each vector to the right node. We issue the actual sends
    // themselves later on.
    const auto cdb_size = cdb.size();

    // The hash table must be empty.
    assert(hash_table.hash_table_size() == 0);
    unsigned ours{};

    for (unsigned i = 0; i < cdb_size; i++) {
      auto &cde = cdb[i];
      auto &dbe = db[cde.i];
      const auto owner = slot_map[UidHashTable::get_slot(dbe.uid)];
      if (owner != rank) {
        outgoing_vectors[owner].emplace_back(i);
        // We've already erased everything at the call site.
      } else {
        // Add back into the table.
        hash_table.insert_uid(dbe.uid);
        ++ours;
      }
    }

    assert(hash_table.hash_table_size() == ours);

    const auto comm_size = outgoing_vectors.size();
    assert(outgoing_sizes.size() == comm_size);
    for (unsigned i = 0; i < static_cast<unsigned>(comm_size); i++) {
      outgoing_sizes[i] = outgoing_vectors[i].size();
    }

    return outgoing_sizes;
  }

  void recompute_initial_received(unsigned i, uint64_t pos,
                                  std::vector<Entry> &tmp_sends,
                                  std::vector<unsigned> &modified_ces) {
    assert(i != rank);
    auto &vec = outgoing_vectors[i];
    auto iter = vec.begin();
    for (; iter < vec.begin() + outgoing_sizes[i]; iter++) {
      recompute_func(*iter);
    }

    // Now prepare any sends that we have left to go out.
    for (unsigned j = 0; iter < vec.end(); iter++, j++) {
      tmp_sends[pos + j] = db[cdb[*iter].i];
      modified_ces.emplace_back(*iter);
    }
  }

  void recompute_extra_received(unsigned i, uint64_t &curr_elem,
                                const uint64_t pos,
                                const std::vector<Entry> &tmp_sends,
                                const std::vector<unsigned> &modified_ces,
                                const size_t outgoing_size,
                                const size_t extra_sends) {
    if (outgoing_vectors[i].size() == outgoing_size) {
      for (unsigned j = 0; j < extra_sends; j++) {
        auto index = modified_ces[curr_elem++];
        db[cdb[index].i] = tmp_sends[pos + j];
        recompute_func(index);
      }
    }
  }

  bool should_receive_more(const unsigned i, const size_t outgoing_size) {
    return outgoing_vectors[i].size() == outgoing_size;
  }

  template <typename F> void apply_filter(F &&filter_func) {
    filter_func(duplicates);
    filter_func(incoming);
  }

  std::unordered_set<unsigned> get_used_set() {
    std::unordered_set<unsigned> used_set(incoming.cbegin(), incoming.cend());
    used_set.insert(duplicates.cbegin(), duplicates.cend());
    return used_set;
  }

  void init_for_removal(const unsigned) {}

  void postprocess_swap(const unsigned, const unsigned) {}

  MPI_Datatype get_entry_type(const unsigned n) {
    return Layouts::get_entry_vector_layout_x_only(n);
  }

private:
  MPI_Datatype get_initial_layout_type(const unsigned i, const unsigned n,
                                       std::vector<int> &offsets) {
    // Warning: this code requires some care!
    // Essentially, the indices in the outgoing_vectors are indices into
    // the CDB. This is not actually exactly what we want: we would like
    // to be able to index based on the indices of those elements in the
    // CDB (i.e. we want to send db[cdb[i].i], not db[i]). This requires
    // us to use the dedicated functionality in the layouts namespace for
    // this.
    // Please also see the WARNING in swap_initial for this.
    // Note that this function returns the vector type for x only, because of
    // the extents issue that's common in MPI.
    return Layouts::get_entry_layout_from_cdb_x_only(
        cdb, outgoing_vectors[i].cbegin(),
        outgoing_vectors[i].cbegin() + outgoing_sizes[i], n, offsets);
  }

public:
  void swap_initial(const unsigned n, MPI_Comm comm) {
    // Each set of data is represented as a separate type.
    // To allow us to later shrink each of the outgoing_vectors quickly, we
    // index from the front when creating the types.
    const auto comm_size = outgoing_sizes.size();

    // WARNING: this code _does not_ respect your choices when it comes to the
    // way that you want to send entries. This is because there seems to be a
    // bug inside MPI_Alltoallw across multiple libraries that, when executing
    // sends in this way, appears to get confused when the data that's being
    // sent is non-contiguous. You'll see this (if you're lucky) in the debug
    // output for your library, but essentially the libraries seem to think that
    // the type signatures across nodes do not match. This is a blatant lie (by
    // definition, we've made sure they're identical), but it's a bug that isn't
    // in our control.
    //
    // Thus, this portion of the code simply uses the coefficient representation
    // regardless, because that seems to work consistently. You've been warned!
    std::vector<MPI_Datatype> types(comm_size);
    {
      // This is a temporary vector: we use this to save on repeated
      // allocations in the calls to the layouts constructor below.
      std::vector<int> offsets;

      for (unsigned i = 0; i < static_cast<unsigned>(comm_size); i++) {
        if (i == static_cast<unsigned>(rank)) {
          // Apparently supplying MPI_DATATYPE_NULL (even though we never send
          // anything to ourselves) isn't allowed by most MPI implementations.
          // We just set this instead.
          types[i] = MPI_CHAR;
          continue;
        }

        types[i] = get_initial_layout_type(i, n, offsets);
      }
    }

    // Now we just carry out the AlltoAllw. Because we deal with all of the
    // complexity via typing, establishing everything we need is actually
    // quite straightforward.
    std::vector<int> rcounts(comm_size, 1);
    // We don't send any to ourselves.
    rcounts[rank] = 0;

    // The displacements are all encoded in the types, so this is essentially
    // all we need.
    std::vector<int> rdispls(comm_size, 0);

    // WARNING WARNING WARNING: this code works on OpenMPI but for a very
    // particular reason. Essentially, OpenMPI uses pipelining by default in
    // order to make operations faster, but this comes at the cost of
    // correctness when different types are supplied by the sender vs the
    // receiver. This is not supposed to be the case in MPI more generally, as
    // the types are actually just supposed to follow the same type map.
    // However, because in this case the swaps are essentially 1:1, the
    // pipelining doesn't seem to fail here. This is likely also because of
    // the use of MPI_IN_PLACE, which likely prevents the underlying
    // implementation from doing certain things. This bug is highlighted in
    // more detail in https://github.com/open-mpi/ompi/issues/134 and
    // https://github.com/open-mpi/ompi/issues/11193.
    // On the other hand, this code should always work for MPICH.
    MPI_Alltoallw(MPI_IN_PLACE, nullptr, nullptr, nullptr, db.data(),
                  rcounts.data(), rdispls.data(), types.data(), comm);

    // Free the types.
    for (unsigned i = 0; i < comm_size; i++) {
      if (i == static_cast<unsigned>(rank)) {
        continue;
      }

      MPI_Type_free(&types[i]);
    }
  }

  std::vector<Entry> &get_db() { return db; }
  std::vector<CompressedEntry> &get_cdb() { return cdb; }
  void do_extra(MPI_Comm) {}

  static constexpr auto TYPE_TAG = ContextChange::REDIST_DB;
  static constexpr auto SEND_TAG = 42;

private:
  UidHashTable &hash_table;
  const std::array<int, DB_UID_SPLIT> &slot_map;
  std::vector<CompressedEntry> &cdb;
  std::vector<Entry> &db;
  std::vector<unsigned> &duplicates;
  std::vector<unsigned> &incoming;
  std::vector<std::vector<unsigned>> outgoing_vectors;
  std::vector<uint64_t> outgoing_sizes;
  unsigned rank;
};

template <typename T>
static void redistribute_database_impl(T &alg, const unsigned n,
                                       MPI_Comm comm) {
  // This function is a bit tricky to understand. Essentially, we need to send
  // all of the entries from each process to each other process that should
  // belong there, but without using too much extra memory. We also can't
  // overwrite our entries that haven't been sent yet. This essentially means
  // we have a distributed instance of a scheduling problem, which is... fun.

  // This function is written to make sure that the distribution happens
  // efficiently. Briefly:
  // 1. Each process first works out how many vectors this process holds that
  // belong to each other process. This information is then globally collated.
  // 2. Each process works out the pairwise minimum across each link. You can
  // view this as
  //    computing n_ij = min(nr(i, j), nr(j, i)), where nr(x, y) is the number
  //    of vectors sent from process x to process y.
  // 3. The entire communicator enters a collective call (MPI_Alltoallw) that
  // essentially swaps n_ij vectors between n_ij for all i, j. This is done
  // for
  //    efficiency reasons.
  // 4. Finally, each process issues the remaining sends and receives to tidy
  // up any leftover vectors.
  // Note that this stage of the algorithm _does_ use more memory: we used to
  // issue these one-by-one, but in higher dimensions this can cause internal
  // MPI resources to run out due to the high number of outstanding requests.
  // Thus, we pack these into individual buffers and use an AlltoAllv, before
  // copying the rest over.
  //
  // This function has some quirks:
  // 1. We only issue sends and receives that are unblocking. This is to
  // prevent deadlocks.
  // 2. We only do any database resizing after this process has finished
  // sending and receiving all entries, because we need stable addressing for
  // unblocking MPI sends to work. Please note that the receiver needs to do
  // quite a bit of work to get all of this to line up: we can end up with
  // sends that are either erroneous or just a little bit useless. This is
  // somewhat unavoidable.

  // The first stage of this algorithm is to make redistribution decisions.
  // We'll make these bucketing decisions exactly once (as mentioned above) to
  // tackle the thread safety issues we're likely to encounter. We produce
  // unique compressed databases for each node so that we can use a collective
  // call later for doing the bulk of the transfer.
  int comm_size;
  int rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &rank);

  assert(rank >= 0);
  assert(comm_size >= 1);

  auto &outgoing_sizes = alg.get_outgoing_sizes();
  std::vector<uint64_t> extra_sends(comm_size, 0);

  const auto db_info =
      calculate_redist_traffic<T::TYPE_TAG>(outgoing_sizes, rank, comm);
  TRACK(T::TYPE_TAG, db_info.outgoing_total * Layouts::get_entry_size(n));

  // We also allow each algorithm to carry out some extra work if they need
  // to.
  alg.do_extra(comm);

  auto outgoing_total = db_info.outgoing_total;
  // As an optimisation, we're now going to work out how much we will send in
  // the initial batch. This is work that we do to minimise the number of MPI
  // requests we issue in total.
  std::vector<unsigned> modified_ces;

  // This vector will come in useful later.
  std::vector<Entry> tmp_sends;

  // These variables are used to track the number of incoming and
  // outgoing "extra sends" that we issue.
  uint64_t nr_incoming{};
  uint64_t nr_outgoing{};

  {
    // Work out how much we'll send in the initial batch.
    for (unsigned i = 0; i < static_cast<unsigned>(comm_size); i++) {
      // N.B This covers the fact that we never send any data to ourselves.
      if (i == static_cast<unsigned>(rank)) {
        extra_sends[i] = 0;
        outgoing_sizes[i] = 0;
        continue;
      }

      const auto tmp_send =
          std::min(outgoing_sizes[i], db_info.incoming_sizes[i]);
      // This contains the "extra" vectors we'll send i.e those that are not
      // covered in the initial send.
      extra_sends[i] =
          std::max(outgoing_sizes[i], db_info.incoming_sizes[i]) - tmp_send;

      // We keep this to allow us to deal with any extra outgoings that we
      // might have later.
      outgoing_sizes[i] = tmp_send;
      outgoing_total -= tmp_send;
    }

    // We'll reserve the number of modified ces ahead of time.
    modified_ces.reserve(
        std::accumulate(extra_sends.cbegin(), extra_sends.cend(), uint64_t(0)));

    // Each algorithm kind deals with this particular operation in their own
    // manner. Because there's some some implementation choices here, we briefly
    // explicate them:
    // 1) The use of MPI_IN_PLACE. It's possible to ask for MPI
    // to overwrite sent entries directly, which means we don't need to allocate
    // extra storage for vectors. This comes at the cost of typically requiring
    // many much smaller messages to be sent, rather than a large contiguous
    // buffer. In practice, the use of MPI_IN_PLACE appears to introduce a
    // factor of 2 slowdown: this is likely because the usage of MPI_IN_PLACE
    // seems to introduce a blocking point (i.e A must send all entries to B,
    // before B sends all their entries to A: we might want to do this in full
    // duplex instead). In some settings this seems not to matter (cf. database
    // redistribution): however, for repeated calls into this function (cf.
    // BDGL-style redistribution) this is very expensive (likely due to
    // latency). In these settings, it may be wiser to pack and unpack these
    // sends directly, rather than using MPI_IN_PLACE without these guards.
    //
    // 2) If MPI_IN_PLACE is not used, you'll need to consider how much
    // temporary storage you do want to allocate. Simply copying everything into
    // buffers likely to be inviable due to the amount of storage needed. On the
    // other hand, "pulsing" the sends will introduce extra overhead. We
    // recommend that you measure an appropriate buffer size for your network by
    // varying the already existing size in the BDGL function.
    //
    // 3) The underlying data type that's used. This is actually more of an MPI
    // issue than not,
    //    but if your implementation uses MPI_Alltoallw with MPI_IN_PLACE it
    //    appears that most libraries trip over themselves and fail to match the
    //    type signatures properly, which is a pain. This is less of an issue in
    //    the non-initial swaps, but to keep everything consistent at the higher
    //    layers we recommend that you use the same serialisation format for
    //    both the initial and non-initial sends. This will change depending on
    //    which algorithm you're using, so the algorithm should provide a way to
    //    find out about that type.
    // NOTE: the type itself here needs to be a vector type (i.e resized to the
    // right extent), but we don't explicitly write that in the function name
    // because it implies that there's a get_entry_type function too, when there
    // really isn't.
    auto type = alg.get_entry_type(n);

    // Anyway, we let each underlying implementation deal with this.
    alg.swap_initial(n, comm);

    // Now we allocate the extra space that we need for sending / receiving
    // the other entries. This works by allocating exactly as much space as we
    // need for sending / receiving entries from each other process.
    tmp_sends.resize(
        std::accumulate(extra_sends.cbegin(), extra_sends.cend(), uint64_t(0)));

    // We count how many we've sent so that the offsets work properly.
    uint64_t pos{};

    // We now deal with those that we've already sent, and then re-pack the
    // existing entries.
    std::vector<MPI_Request> outgoing_reqs(comm_size, MPI_REQUEST_NULL);

    for (unsigned i = 0; i < static_cast<unsigned>(comm_size); i++) {
      // No need to do anything for ourselves.
      if (i == static_cast<unsigned>(rank))
        continue;

      // Otherwise, let "the algorithm" do its work.
      alg.recompute_initial_received(i, pos, tmp_sends, modified_ces);

      // We now send out the rest of the data. This is done by issuing a
      // series of non-blocking sends or receives, depending on if we have
      // data coming in or not. This also allows us to just send the data and
      // hope for the best.

      // We first check if we're actually expecting to receive something. If
      // we are, then we can just issue a send. This exists to make sure that
      // everything lines up properly in the rare case that all of the sends
      // were handled above.
      if (extra_sends[i] == 0) {
        continue;
      }

      if (alg.should_receive_more(i, outgoing_sizes[i])) {
        // If we've sent all of ours, then we're receiving more.
        MPI_Irecv(&tmp_sends[pos], extra_sends[i], type, i, T::SEND_TAG, comm,
                  &outgoing_reqs[i]);
        nr_incoming += extra_sends[i];
      } else {
        // We're sending.
        MPI_Isend(&tmp_sends[pos], extra_sends[i], type, i, T::SEND_TAG, comm,
                  &outgoing_reqs[i]);
        nr_outgoing += extra_sends[i];
      }

      pos += extra_sends[i];
    }

    // Wait for all of the extra sends / receives to finish.
    MPI_Waitall(outgoing_reqs.size(), outgoing_reqs.data(),
                MPI_STATUSES_IGNORE);
    // Now we can cleanup all of the stuff that's come in.
    MPI_Type_free(&type);
  }

  // Now we've done all of the sends etc we can resize the database, since
  // it's all been finished.
  auto &db = alg.get_db();
  auto &cdb = alg.get_cdb();

  const auto old_size = db.size();
  // This is exactly the expected size of our new DB.
  const auto new_size = old_size - nr_outgoing + nr_incoming;
  // First we'll grow the db, if needed. We can do this because the various
  // sends must have finished already, and they belong to separate buffers
  // anyway. That is, we will not disrupt any MPI operations by growing here.

  if (new_size > old_size) {
    db.resize(new_size);
    cdb.resize(new_size);

    // The generated code is better for this vs emplacing back.
    auto insert_pos = modified_ces.size();
    modified_ces.resize(insert_pos + (new_size - old_size));
    for (unsigned i = old_size; i < new_size; i++) {
      cdb[i].i = i;
      modified_ces[insert_pos++] = i;
    }
  }

  // The exact number of modified ces that we have varies on the extra sends
  // we've done. In either case, it's the maximum of nr_outgoing (because we
  // will have written these in earlier) or nr_incoming (because we will have
  // grown to this length above).
  assert(modified_ces.size() == std::max(nr_outgoing, nr_incoming));

  // Now we re-initialise those elements that have come in.
  uint64_t curr_elem{};
  uint64_t pos{};

  for (unsigned i = 0; i < static_cast<unsigned>(comm_size); i++) {
    if (i == static_cast<unsigned>(rank)) {
      continue;
    }

    // Recompute the extra information.
    alg.recompute_extra_received(i, curr_elem, pos, tmp_sends, modified_ces,
                                 outgoing_sizes[i], extra_sends[i]);

    // Move `pos` forward appropriately.
    // As before, we always need to do this.
    pos += extra_sends[i];
  }

  // The only thing left to do is to adjust our database sizes.
  // If we've shrunk our database, then we need to be careful here: it's very
  // possible that we've written some entries into memory that won't survive
  // the shrinkage. In practice because we're shrinking this won't lead to
  // segfaults (although it may), but it's undefined behaviour and practically
  // slows the sieve down a lot. To circumvent that, we have to move some
  // things around.
  if (new_size < old_size) {
    // Each algorithm can store the ones they've used as they see fit.
    auto used = alg.get_used_set();

    // Some algorithms may need to do more work for post-processing purposes.
    alg.init_for_removal(new_size);
    // This split is to allow us to keep track of the compressed entries we
    // haven't used, along with knowing whether they're valid candidates or
    // not.
    std::vector<unsigned> unused;
    std::unordered_set<unsigned> unused_set;

    unused.reserve(old_size - new_size);
    unused_set.reserve(old_size - new_size);

    for (const auto v : modified_ces) {
      if (used.count(v) == 0) {
        unused_set.insert(v);
        if (v < new_size) {
          unused.emplace_back(v);
        }
      }
    }

    // We should have exactly this many entries unused.
    assert(unused_set.size() == old_size - new_size);

    // This lambda is used to swap any compressed entries we've read to into
    // the saved portion of the database. Essentially, we iterate over `input`
    // and swap any compressed entries we'd lose into the retained portion of
    // the cdb.

    const auto filter_ces = [&used, &unused, &alg,
                             new_size](std::vector<unsigned> &input) {
      auto &cdb = alg.get_cdb();
      for (auto &v : input) {
        if (v < new_size)
          continue;
        assert(!unused.empty());
        const auto old_val = v;
        const auto new_val = unused.back();

        std::swap(cdb[v], cdb[unused.back()]);
        std::swap(v, unused.back());
        unused.pop_back();
        alg.postprocess_swap(old_val, new_val);

        // This is important: this insertion tracks that we've already
        // swapped `v`. This is useful because it means we don't have to
        // swap it again later.
        used.insert(v);
      }
    };

    // Filter out those that we've used.
    alg.apply_filter(filter_ces);

    // Now we have to deal with the compressed entries we had originally. The
    // idea here is similar to in filter_ces, but we only swap those entries
    // we haven't already used.
    for (unsigned i = new_size; i < old_size; i++) {
      // Don't count those we've already used.
      if (used.count(i) != 0 || unused_set.count(i) != 0)
        continue;
      assert(!unused.empty());
      std::swap(cdb[i], cdb[unused.back()]);
      alg.postprocess_swap(i, unused.back());
      unused.pop_back();
      used.insert(i);
    }

    assert(unused.empty());

    // Now all of the unused compressed entries now live in [new_size,
    // old_size). Now we need to swap over some db elements. The idea here is
    // similar to before: if there's any compressed elements in [new_size,
    // old_size) that point into the retained part of the db, then there must
    // be as many elements in the db that point into the part we'll discard.
    // By finding those that point into the retained portion, we can swap out.
    std::vector<unsigned> points_inside;
    for (unsigned i = new_size; i < old_size; i++) {
      if (cdb[i].i < new_size) {
        points_inside.emplace_back(i);
      }
    }

    if (points_inside.size() != 0) {
      // Only go over the portion we'll retain.
      for (unsigned i = 0; i < new_size; i++) {
        if (cdb[i].i < new_size) {
          continue;
        }

        // Same idea as in filter_ces: we just swap over the memory and the
        // indices.
        assert(!points_inside.empty());
        auto &ce1 = cdb[i];
        auto &ce2 = cdb[points_inside.back()];
        std::swap(db[ce1.i], db[ce2.i]);

        // Everything is updated at the callsite.
        std::swap(ce1.i, ce2.i);
        alg.postprocess_swap(i, points_inside.back());
        points_inside.pop_back();
      }

      // This must be true.
      assert(points_inside.empty());
    }

    // Finally resize them.
    db.resize(new_size);
    cdb.resize(new_size);
  }
  // There's no "else" case here because we will have handled it in the
  // initial growth.
}

std::vector<unsigned> MPIWrapper::bdgl_exchange(
    std::vector<Entry> &db, std::vector<CompressedEntry> &cdb, const unsigned n,
    std::vector<size_t> &sizes, std::vector<uint32_t> &buckets,
    const size_t bsize, const std::vector<uint32_t> &owner,
    const std::vector<uint32_t> &ours, MPI_Comm comm) {
  // The real heavy lifting is done in the implementation function.
  int comm_size;
  int rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);
  RedistBDGLAlg alg(cdb, db, sizes, buckets, bsize, owner, ours, comm_size,
                    rank);

  redistribute_database_impl(alg, n, comm);
  return alg.get_used();
}

void MPIWrapper::redistribute_database(
    std::vector<Entry> &db, std::vector<CompressedEntry> &cdb,
    const std::array<int, DB_UID_SPLIT> &slot_map, const unsigned n,
    UidHashTable &hash_table, MPI_Comm comm, std::vector<unsigned> &duplicates,
    std::vector<unsigned> &incoming) noexcept {

  // The real heavy lifting is done in the implementation function.
  int comm_size;
  int rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  RedistDBAlg alg(hash_table, slot_map, cdb, db, duplicates, incoming,
                  static_cast<unsigned>(comm_size),
                  static_cast<unsigned>(rank));
  redistribute_database_impl(alg, n, comm);
}

void MPIWrapper::grab_lift_bounds(std::vector<FT> &lift_bounds,
                                  FT &lift_max_bound, MPI_Comm comm) noexcept {

  // This function saves an MPI call at the cost of two extra (small)
  // allocations.
  auto copy = lift_bounds;
  copy.emplace_back(lift_max_bound);
  auto size = copy.size();

  TRACK(ContextChange::LIFT_BOUNDS, sizeof(FT) * size);
  MPI_Allreduce(MPI_IN_PLACE, copy.data(), size, Layouts::get_data_type<FT>(),
                MPI_MIN, comm);

  lift_max_bound = copy.back();
  copy.pop_back();
  lift_bounds = copy;
}

void MPIWrapper::forward_and_gather_buckets(
    const unsigned n, const unsigned buckets_per_rank,
    const std::vector<std::vector<std::vector<unsigned>>> &cbuckets,
    const std::vector<Entry> &db, const std::vector<int> &sizes,
    std::vector<ZT> &incoming_buffer, std::vector<ZT> &outgoing_buffer,
    std::vector<bucket_pair> &bucket_pairs, std::vector<int> &scounts,
    std::vector<int> &sdispls, std::vector<int> &rcounts,
    std::vector<int> &rdispls, const MPI_Datatype entry_type, MPI_Comm comm,
    MPI_Request *request) noexcept {

  int rank, nr_ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nr_ranks);
  assert(scounts.size() == unsigned(nr_ranks));
  assert(sdispls.size() == unsigned(nr_ranks));
  assert(rcounts.size() == unsigned(nr_ranks));
  assert(rdispls.size() == unsigned(nr_ranks));

  // The very first thing we do is work out how much storage we need in each
  // direction and copy accordingly. Briefly, most MPI implementations seem to
  // practically perform better when data can be batched and sent in a
  // contiguous buffer. This appears to be because non-contiguous sends seem
  // to follow the MPI mantra of using little extra memory: each bucket is
  // sent individually after being copied into a smallish buffer. This costs
  // us greatly, because: 1) Each bucket is re-collected using a series of
  // random accesses, which is not cheap. 2) Some MPI implementations seem to
  // cache data types even after freeing, increasing memory usage. 3) Each
  // bucket is sent in a single message, rather than batching messages up.
  // This increases latency
  //    massively.
  //
  // Points 2 and 3 are the major costs involved in this entire venture, which
  // is somewhat surprising: the memory allocated by MPI implementations (both
  // OpenMPI and MPICH) can be (in some situations) multiple GB for its
  // internal buffers. There does not seem to be much re-use inside MPI, so
  // using individual types is rather stymying.
  //
  // Thus, we have to pack both sent data and received data into a single
  // array. In the outgoing case, this is easy: we just copy the data over. We
  // know exactly how many entries we will send, since that corresponds to the
  // size of each outgoing vector. The incoming case requires more care.

  // DESIGN NOTES: please note that the displacements etc for MPI_Ialltoallv
  // are expressed in terms of the number of entry_types that are being sent,
  // not the number of ZTs that are sent. In practice, the restriction to int
  // displacements is not an issue: as each bucket is roughly sqrt(db_size) in
  // size we would need the dimension to be rather large for this to be
  // violated (napkin maths says d ~= 210).
  //
  // Another design note: this code deliberately does not pack the outgoing
  // elements in any way to reduce the bandwidth usage. This is because (as
  // n->\infinity) the number of duplicate elements across multiple buckets
  // tends to 0. Indeed, if you add a std::unordered_set to insertion loop
  // below, you'll see that by dimension 87 the uniqueness ratio is around
  // 98%.

  int nr_outgoing{};
  for (unsigned i = 0; i < unsigned(nr_ranks); i++) {
    if (i == static_cast<unsigned>(rank))
      continue;

    // Count the number of outgoing buckets.
    const auto &bucket_set = cbuckets[i];
    const auto size =
        std::accumulate(bucket_set.cbegin(), bucket_set.cend(), 0,
                        [](const int lhs, const std::vector<unsigned> &bucket) {
                          return lhs + bucket.size();
                        });

    scounts[i] = size;
    sdispls[i] = nr_outgoing;
    nr_outgoing += size;
  }

  // Now resize. Note: the clear is to prevent many extra needless memmoves.
  outgoing_buffer.clear();

  // WARNING: this was a really irritating bug to diagnose, so it's in a big
  // block. Essentially, certain MPI implementations will complain if you supply
  // a null pointer to a function for a receive buffer, even if the send count
  // is 0. This is a little bit annoying. We circumvent this by making a small
  // allocation if absolutely necessary, although this is irritating.
  // This bug only ever seems to manifest in when the sieve is used in BDGL
  // mode: this is because, in the BDGL sieve, certain nodes will finish their
  // work early and thus just end up sending buckets out. The converse of this
  // is that some nodes will receive no buckets in certain iterations, and so
  // (especially in settings with few nodes) outgoing_buffer might have size 0.
  outgoing_buffer.resize(std::max<unsigned>(1, n * nr_outgoing));

  // The incoming one is also straightforward: we just sum the sizes in
  // `sizes`. N.B To take advantage of this, we store this for later.
  int offset{};

  // This loop also does the very important task of calculating exactly which
  // positions in the incoming buffer correspond to each bucket. This is used
  // for reconstruction later, and so it has to be done here. Note: positions
  // and the count are also in terms of the number of buckets, and not in
  // terms of the number of elements.

  for (unsigned i = 0; i < unsigned(nr_ranks); i++) {
    if (i == static_cast<unsigned>(rank))
      continue;

    // Add up the number of incoming buckets.
    int total{};
    for (unsigned j = 0; j < buckets_per_rank; j++) {
      bucket_pairs[j * nr_ranks + i].size = sizes[i * buckets_per_rank + j];
      bucket_pairs[j * nr_ranks + i].pos = offset + total;
      total += sizes[i * buckets_per_rank + j];
    }

    // This counts how many entry types we'll receive.
    rcounts[i] = total;
    rdispls[i] = offset;
    offset += total;
  }

  // Offset contains how many elements we'll receive, so multiply by `n` to
  // adjust accordingly.
  incoming_buffer.clear();
  // Same quirk as above, although this seems far less likely to be triggered
  // than the outgoing case.
  incoming_buffer.resize(std::max<unsigned>(1, offset * n));
  assert(incoming_buffer.size() != 0);
  assert(incoming_buffer.data() != nullptr);
  if (incoming_buffer.data() == nullptr) {
    std::cerr << "[Rank " << rank
              << "] Error: incoming_buffer.data() is a nullptr" << std::endl;
  }

  // Now we copy over into the outgoing buffer.
  // Here we walk through the database and copy over.
  unsigned curr{};
  for (unsigned i = 0; i < unsigned(nr_ranks); i++) {
    if (i == static_cast<unsigned>(rank))
      continue;

    for (const auto &bucket_set : cbuckets[i]) {
      for (const auto &index : bucket_set) {
        std::copy(db[index].x.cbegin(), db[index].x.cbegin() + n,
                  outgoing_buffer.begin() + curr);
        curr += n;
      }
    }
  }

  // N.B This assert also covers the case that we have nothing to send.
  assert(curr == outgoing_buffer.size() ||
         n * nr_outgoing == 0 && curr == 0 && outgoing_buffer.size() == 1);

  // Now we can issue the send.
  // N.B The subtraction here is just meant to adjust the case that we aren't
  // sending any data: it prevents an off-by-one.
  TRACK(ContextChange::BGJ1_BUCKET_SEND,
        (outgoing_buffer.size() - (n * nr_outgoing == 0)) * sizeof(ZT));
  MPI_Ialltoallv(outgoing_buffer.data(), scounts.data(), sdispls.data(),
                 entry_type, incoming_buffer.data(), rcounts.data(),
                 rdispls.data(), entry_type, comm, request);
}

template <ContextChange state, typename RT>
RT static get_reduced_value(MPI_Comm comm, RT value) noexcept {
  static_assert(state == ContextChange::GET_CPU_TIME ||
                state == ContextChange::GET_EXTRA_MEMORY);

  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == MPIWrapper::global_root_rank) {
    broadcast_context_change(state, 0, comm);
  }

  // We need to adjust the send-recv buffer depending on whether the caller
  // is the root rank or not (because MPI_IN_PLACE can only be used on the
  // root rank).
  const auto *send_buffer =
      (rank == MPIWrapper::global_root_rank) ? MPI_IN_PLACE : &value;
  auto *recv_buffer = (rank == MPIWrapper::global_root_rank) ? &value : nullptr;

  MPI_Reduce(send_buffer, recv_buffer, 1, Layouts::get_data_type<RT>(), MPI_SUM,
             MPIWrapper::global_root_rank, comm);
  TRACK(state, sizeof(value));
  return value;
}

long double MPIWrapper::get_cpu_time(MPI_Comm comm,
                                     std::clock_t time) noexcept {

  const auto time_out = 1000.0 * time / CLOCKS_PER_SEC;
  return get_reduced_value<ContextChange::GET_CPU_TIME>(comm, time_out);
}

uint64_t MPIWrapper::get_extra_memory_used(MPI_Comm comm,
                                           uint64_t extra_used) noexcept {
  return get_reduced_value<ContextChange::GET_EXTRA_MEMORY>(comm, extra_used);
}

template <ContextChange state, typename F>
static uint64_t get_total(MPI_Comm comm, F &&func,
                          ContextChange type = ContextChange::LAST) noexcept {
#ifndef MPI_TRACK_BANDWIDTH
  return 0;
#else
  constexpr bool is_total = state == ContextChange::TOTAL_MESSAGES ||
                            state == ContextChange::TOTAL_BANDWIDTH;

  constexpr bool is_individual = state == ContextChange::MESSAGES_FOR ||
                                 state == ContextChange::BANDWIDTH_FOR;

  static_assert(is_individual || is_total);

  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == MPIWrapper::global_root_rank) {
    broadcast_context_change(state, static_cast<unsigned>(type), comm);
  }

  const auto counts = tracker.get_counts();
  auto total = func(counts, type);

  // We need to adjust the send-recv buffer depending on whether the caller
  // is the root rank or not (because MPI_IN_PLACE can only be used on the
  // root rank).
  const auto *send_buffer =
      (rank == MPIWrapper::global_root_rank) ? MPI_IN_PLACE : &total;
  auto *recv_buffer = (rank == MPIWrapper::global_root_rank) ? &total : nullptr;
  MPI_Reduce(send_buffer, recv_buffer, 1,
             Layouts::get_data_type<decltype(total)>(), MPI_SUM,
             MPIWrapper::global_root_rank, comm);

  if (rank == MPIWrapper::global_root_rank) {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    if (is_total) {
      const auto extra = sizeof(uint64_t) * (comm_size - 1);
      TRACK(state, extra);
      total += extra;
    } else {
      TRACK(state, 0);
    }
  }
  return total;
#endif
}

uint64_t MPIWrapper::get_total_bandwidth(MPI_Comm comm) noexcept {
#ifndef MPI_TRACK_BANDWIDTH
  (void)comm;
  return 0;
#else
  using CountType = decltype(tracker.get_counts());

  auto total_bandwidth_func = [](const CountType counters,
                                 const ContextChange) {
    uint64_t total{};
    for (const auto &entry : counters) {
      total += entry.data;
    }
    return total;
  };

  return get_total<ContextChange::TOTAL_BANDWIDTH>(comm, total_bandwidth_func);
#endif
}

uint64_t MPIWrapper::get_total_messages(MPI_Comm comm) noexcept {
#ifndef MPI_TRACK_BANDWIDTH
  (void)comm;
  return 0;
#else
  using CountType = decltype(tracker.get_counts());
  auto total_messages_func = [](const CountType counters, const ContextChange) {
    uint64_t total{};
    for (const auto &entry : counters) {
      total += entry.count;
    }
    return total;
  };

  return get_total<ContextChange::TOTAL_MESSAGES>(comm, total_messages_func);
#endif
}

uint64_t MPIWrapper::get_messages_for(const ContextChange type,
                                      MPI_Comm comm) noexcept {
#ifndef MPI_TRACK_BANDWIDTH
  (void)type;
  (void)comm;
  return 0;
#else
  using CountType = decltype(tracker.get_counts());
  auto individual_message_func = [](const CountType counters,
                                    const ContextChange type) {
    assert(type != ContextChange::LAST);
    return counters[static_cast<unsigned>(type)].count;
  };
  return get_total<ContextChange::MESSAGES_FOR>(comm, individual_message_func,
                                                type);
#endif
}

uint64_t MPIWrapper::get_bandwidth_for(const ContextChange type,
                                       MPI_Comm comm) noexcept {
#ifndef MPI_TRACK_BANDWIDTH
  (void)type;
  (void)comm;
  return 0;
#else
  using CountType = decltype(tracker.get_counts());
  auto individual_message_func = [](const CountType counters,
                                    const ContextChange type) {
    assert(type != ContextChange::LAST);
    return counters[static_cast<unsigned>(type)].data;
  };
  return get_total<ContextChange::BANDWIDTH_FOR>(comm, individual_message_func,
                                                 type);
#endif
}

std::array<uint64_t, 2> MPIWrapper::get_unique_ratio(const uint64_t nr_uniques,
                                                     const uint64_t nr_sends,
                                                     MPI_Comm comm) noexcept {

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == MPIWrapper::global_root_rank) {
    broadcast_context_change(ContextChange::GET_UNIQUE_RATIO, 0, comm);
  }

  std::array<uint64_t, 2> arr{nr_uniques, nr_sends};
  auto *send_buf =
      (rank == MPIWrapper::global_root_rank) ? MPI_IN_PLACE : arr.data();
  auto *recv_buf =
      (rank == MPIWrapper::global_root_rank) ? arr.data() : nullptr;

  MPI_Reduce(send_buf, recv_buf, 2, MPI_UINT64_T, MPI_SUM,
             MPIWrapper::global_root_rank, comm);
  return arr;
}

uint64_t MPIWrapper::get_adjust_timings(const uint64_t time,
                                        MPI_Comm comm) noexcept {
  uint64_t out{};
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == MPIWrapper::global_root_rank) {
    broadcast_context_change(ContextChange::GET_ADJUST_TIMINGS, 0, comm);
  }

  MPI_Reduce(&time, &out, 1, MPI_UINT64_T, MPI_SUM,
             MPIWrapper::global_root_rank, comm);
  return out;
}

void MPIWrapper::reset_bandwidth(MPI_Comm comm) noexcept {
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == MPIWrapper::global_root_rank) {
    broadcast_context_change(ContextChange::RESET_BANDWIDTH, 0, comm);
  }
#ifdef MPI_TRACK_BANDWIDTH
  tracker.clear_counts();
#endif
}

void MPIWrapper::start_bdgl(const size_t nr_buckets_aim, const size_t blocks,
                            const size_t multi_hash, MPI_Comm comm) noexcept {
  static_assert(std::numeric_limits<uint64_t>::max() >=
                std::numeric_limits<size_t>::max());
  std::array<uint64_t, 3> vals{nr_buckets_aim, blocks, multi_hash};
  broadcast_context_change(ContextChange::BDGL, 0, comm);
  MPI_Bcast(vals.data(), vals.size(), MPI_UINT64_T,
            MPIWrapper::global_root_rank, comm);
}

std::array<uint64_t, 3>
MPIWrapper::receive_bdgl_params(MPI_Comm comm) noexcept {
  std::array<uint64_t, 3> vals;
  MPI_Bcast(vals.data(), vals.size(), MPI_UINT64_T,
            MPIWrapper::global_root_rank, comm);
  return vals;
}
