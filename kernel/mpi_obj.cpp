#include "mpi_obj.hpp"
#include "siever.h" // We need this for G6K things.
#include <iostream>
#include <numeric>

// We'll only include these if we're doing MPI things
#ifdef G6K_MPI
#ifndef DOCTEST_CONFIG_DISABLE
#include "doctest/extensions/doctest_mpi.h"
#endif
#include "layouts.hpp"
#include "mpi_cast.hpp"
#include "mpi_wrapper.hpp"
#endif
#include "QEntry.hpp"
#include "fht_lsh.h"
#include <sys/resource.h>

#ifdef MPI_TRACK_BANDWIDTH
extern MPIBandwidthTracker tracker;
#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("build_mpi_obj", 2) {
  // This test case is primarily to make sure that the MPIObj constructor
  // throws on erroneous values for my_threads and my_memory.
  CHECK_THROWS_WITH_AS(MPIObj(true, test_comm, 0),
                       "Cannot instantiate MPIObj with my_memory == 0",
                       std::invalid_argument);

  CHECK_THROWS_WITH_AS(MPIObj(true, test_comm, 1, 0),
                       "Cannot instantiate MPIObj with my_threads == 0",
                       std::invalid_argument);
}
#endif

#ifdef G6K_MPI
MPIObj::MPIObj(const bool is_g6k, const uint64_t comm_,
               const uint64_t my_threads, DistSieverType topology,
               unsigned scale_factor, const unsigned bucket_batches,
               const unsigned scratch_buffers) MPI_DIST_MAY_THROW
    : MPIObj{
          is_g6k,
          (comm_ == 0 ? MPI_COMM_WORLD : MPI_Cast::uint64_to_mpi_comm(comm_)),
          my_threads,
          topology,
          scale_factor,
          bucket_batches,
          scratch_buffers} {}

MPIObj::MPIObj(const bool is_g6k, const MPI_Comm comm_,
               const uint64_t my_threads, DistSieverType topology,
               unsigned scale_factor, const unsigned batches,
               const unsigned scratch_buffers) MPI_DIST_MAY_THROW
    : comm{comm_},
      rank{},
      buckets_per_rank{my_threads},
      memory_per_rank{},
      total_bucket_count{},
      state{MPIObj::State::DEFAULT},
      active{},
      reduce_best_lifts_op{} {

  // Preconditions.
  THROW_OR_OPTIMISE(my_threads == 0, std::invalid_argument,
                    "Cannot instantiate MPIObj with my_threads == 0");

  // This check is just to make sure we don't initialise MPI
  // More than once. This can happen during testing, but it's unlikely
  // to happen during the lifetime of a normal program.
  int initialised;
  MPI_Initialized(&initialised);
  if (!initialised) {
    int thread_support;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &thread_support);
    if (thread_support < MPI_THREAD_FUNNELED) {
      std::cerr << "Error: The MPI implementation does not support "
                   "MPI_THREAD_FUNNELED"
                << std::endl;
      MPI_Abort(comm, 1);
    }
  }

  // We have to initialise this here at the earliest, or we could potentially
  // initialise it before MPI_Init is called, which is disallowed
  reduce_best_lifts_op = MPIWrapper::get_best_lifts_op();

  // Now we need to check if we're alone. If we are and we're the G6K process
  // then we spawn more workers.
  if (is_g6k) {
    int size;
    MPI_Comm_size(comm, &size);
    if (size == 1) {
      std::cerr << "Error: trying to instantiate MPI sieving but only one "
                   "process was instantiated. "
                << std::endl
                << "Make sure you run mpi sieving with mpirun -n 1 <python "
                   "program> : -n <hosts> dist_siever"
                << std::endl;
      MPI_Abort(comm, 1);
    }
  }

  // If we are a non-G6K process then we just need to check that one of us in
  // the comm is a G6K process: if not, the
  // program has been started as an error.
  const auto is_one_g6k = MPIWrapper::is_one_g6k(is_g6k, comm);
  if (!is_one_g6k) {
    std::cerr << "Aborting MPI process: no G6K process found." << std::endl;
    MPI_Abort(comm, 1);
  }

  // We send the topology here first, as the topological changes may
  // change the ranks of the MPI processes.
  if (is_g6k) {
    MPIWrapper::send_topology(topology, comm);
  } else {
    MPIWrapper::get_topology(topology, comm);
  }

  // Now set up the topology and then reset the ranks.
  this->topology = Topology::build_topology(comm, topology);

  // Update our rank under the comm (it's likely changed).
  MPI_Comm_rank(comm, &rank);
  // Now we'll force our ranks to be in the right order.
  auto tmp_comm =
      MPIWrapper::set_root_rank_to_global_root_rank(rank, is_g6k, comm);

  this->comm = tmp_comm;
  MPI_Comm_rank(comm, &rank);

  // Collect the number of buckets for all ranks.
  this->bucket_batches = batches;
  this->scratch_buffers = scratch_buffers;
  MPIWrapper::gather_buckets(buckets_per_rank, scale_factor,
                             this->bucket_batches, this->scratch_buffers, comm);
  MPIWrapper::collect_memory(memory_per_rank, scale_factor * my_threads, comm);

  // Now we set up our various persistent connections. We retain persistent
  // connections for each outgoing message type.
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  const auto nr_other_ranks = comm_size - 1;

  // This contains all of the relevant information etc for both
  // incoming and outgoing buckets.
  bgj1_buckets = bgj1_bucketing_interface(buckets_per_rank, comm_size,
                                          this->bucket_batches, comm);

  // This contains all but the adhoc MPI requests.
  requests = MPIRequest(comm_size, this->bucket_batches);
  bucket_pairs.resize(this->bucket_batches);
  for (auto &v : bucket_pairs) {
    v.resize(buckets_per_rank * comm_size);
  }

  total_bucket_count = buckets_per_rank * comm_size;

  // Set up the slot map. This works out how we will
  // divide up the database based on the uids of each
  // entry.
  MPIWrapper::setup_slot_map(slot_map, memory_per_rank);

  tmp_insertion_buf.resize(bucket_batches);
  for (auto &v : tmp_insertion_buf) {
    v.buffers = std::move(std::vector<inner_outgoing_vector>(comm_size));
  }

  insertion_vector_bufs.resize(bucket_batches);

  MPI_Comm_dup(comm, &stop_comm);

  stopped = false;
  issued_stop = false;

  sat_drop = 0;
  is_barrier_empty = true;

  sync_header_type = Layouts::get_sync_header_type();

  finished_batches.resize(this->bucket_batches);
  scounts.resize(comm_size);
  sdispls.resize(comm_size);
  rcounts.resize(comm_size);
  rdispls.resize(comm_size);

  nr_real_buckets = std::accumulate(memory_per_rank.cbegin(),
                                    memory_per_rank.cend(), unsigned(0));

  scratch_used =
      std::move(std::vector<padded_atom_unsigned>(this->scratch_buffers));
  scratch_lookup.resize(this->bucket_batches);
  scratch_space.resize(this->scratch_buffers);
  free_scratch.store(this->scratch_buffers);

  wait_head = 0;
  finished_sizes.resize(this->bucket_batches);
  is_used = true;

  to_remove = std::vector<UidManageStruct>(comm_size);
  bdgl_bucket_active = std::vector<padded_atom_unsigned>(this->bucket_batches);

#ifdef MPI_TRACK_UNIQUES
  unique_sends = 0;
  nr_sends = 0;

  bucket_size = 0;
  unique_entries = 0;
#endif

#if defined(MPI_TRACK_UNIQUES) || defined(MPI_TIME) ||                         \
    defined(MPI_BANDWIDTH_TRACK)

  if (is_root()) {
    auto now = std::chrono::system_clock::now();
    auto utc =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
            .count();
    filepath_for_track = "stats_output_" + std::to_string(utc) + ".csv";
    std::ofstream writer(filepath_for_track);
    // Write initial header.
    writer << sub_line << std::endl;
    writer.close();
  }
#endif
}

MPIObj::~MPIObj() noexcept {
  MPI_Op_free(&reduce_best_lifts_op);
  MPI_Type_free(&sync_header_type);
  bgj1_buckets.free_comms();
  MPI_Comm_free(&stop_comm);

#ifndef MPI_DIST_TEST
  if (is_root()) {
    send_stop();
  }

  MPI_Finalize();
#endif
}

MPI_Comm MPIObj::get_comm() noexcept { return comm; }

void MPIObj::set_comm(MPI_Comm comm_) noexcept {
  this->comm = comm_;
  MPI_Comm_rank(comm, &rank);
}

#else
MPIObj::MPIObj(const bool, const uint64_t, const uint64_t, const uint64_t,
               DistSieverType, const std::string &) noexcept {}
#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("get_root_rank", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  CHECK(mp.get_root_rank() == MPIWrapper::global_root_rank);
}
#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("get_topology", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  CHECK(mp.get_topology() == DistSieverType::ShuffleSiever);
}

#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("get_rank", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  CHECK(mp.get_rank() == test_rank);
}
#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("is_root", 2) {
  // Check that the global_root_rank isn't too large.
  int size;
  MPI_Comm_size(test_comm, &size);
  REQUIRE(size > MPIWrapper::global_root_rank);

  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};

  MPI_CHECK(MPIWrapper::global_root_rank, mp.is_root());
  if (test_rank != MPIWrapper::global_root_rank) {
    CHECK(!mp.is_root());
  }
}
#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("threads and memory are set", 2) {
  // This test case just makes sure that the initial setup works.
  uint64_t threads[2]{10, 15};
  uint64_t memory[2]{1, 1500};

  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm,
            memory[test_rank], threads[test_rank]};

  // Check the number of threads and memory is as expected.
  const auto number_of_threads = mp.get_buckets_per_rank();
  // The output should be ordered.
  CHECK(number_of_threads[0] == threads[0]);
  CHECK(number_of_threads[1] == threads[1]);

  // Same for the memory.
  const auto memory_per_rank = mp.get_memory_per_rank();
  REQUIRE(memory_per_rank.size() == 2);
  CHECK(memory_per_rank[0] == memory[0]);
  CHECK(memory_per_rank[1] == memory[1]);
}
#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("send_gso", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  test_comm = mp.get_comm();

  std::array<double, 9> arr{1, 2, 3, 4, 5, 6, 7, 8, 9};
  SUBCASE("throws if mp is not root") {
    if (!mp.is_root()) {
      CHECK_THROWS_WITH_AS(mp.send_gso(3, arr.data()),
                           "cannot call send_gso on a non-root rank",
                           std::logic_error);
    }
  }

  if (mp.is_root()) {
    mp.send_gso(3, arr.data());
  } else {
    std::vector<double> result;
    MPIWrapper::receive_gso(result, test_comm);
    REQUIRE(result.size() == arr.size());
    for (unsigned i = 0; i < arr.size(); i++) {
      CHECK(arr[i] == result[i]);
    }
  }
}
#endif

void MPIObj::send_gso(unsigned int full_n,
                      const double *const mu) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "cannot call send_gso on a non-root rank");
  // NOTE: the squaring here is deliberate.
  MPIWrapper::send_gso(full_n * full_n, mu, comm);
#else
  (void)(full_n);
  (void)(mu);
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("receive_gso", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  test_comm = mp.get_comm();
  std::vector<double> arr{1, 2, 3, 4, 5, 6, 7, 8, 9};

  SUBCASE("cannot receive on a root rank") {
    if (mp.is_root()) {
      std::vector<double> out;
      CHECK_THROWS_WITH_AS(mp.receive_gso(out),
                           "cannot call receive_gso on a root rank",
                           std::logic_error);
    }
  }

  if (mp.is_root()) {
    mp.send_gso(3, arr.data());
  } else {
    std::vector<double> out;
    mp.receive_gso(out);
    CHECK(out == arr);
  }
}
#endif

void MPIObj::receive_gso(std::vector<double> &gso) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(is_root(), std::invalid_argument,
                    "cannot call receive_gso on a root rank");
  MPIWrapper::receive_gso(gso, comm);
#else
  (void)gso;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("receive_gso_no_header", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  test_comm = mp.get_comm();
  std::vector<double> arr{1, 2, 3, 4, 5, 6, 7, 8, 9};
  if (mp.is_root()) {
    mp.send_gso(3, arr.data());
  } else {
    const auto status = mp.receive_command();
    CHECK(status[0] == ContextChange::LOAD_GSO);
    CHECK(status[1] == 9);
    std::vector<double> gso;
    mp.receive_gso_no_header(status[1], gso);
    CHECK(gso == arr);
  }
}
#endif

void MPIObj::receive_gso_no_header(
    const unsigned size, std::vector<double> &gso) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(is_root(), std::logic_error,
                    "cannot call receive_gso_no_header on a root rank");
  MPIWrapper::receive_gso_no_header(size, gso, comm);
#else
  (void)size;
  (void)gso;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("send_status", 2) {
  // Test that sending the status works.
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  test_comm = mp.get_comm();
  SUBCASE("cannot call send_status on non-root rank") {
    if (!mp.is_root()) {
      CHECK_THROWS_WITH_AS(mp.send_status(5),
                           "Cannot call send_status on non-root rank",
                           std::logic_error);
    }
  }

  if (mp.is_root()) {
    mp.send_status(5);
  } else {
    const auto status = MPIWrapper::receive_command(test_comm);
    CHECK(status[0] == ContextChange::CHANGE_STATUS);
    CHECK(status[1] == 5);
  }
}
#endif

void MPIObj::send_status(const unsigned status) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "Cannot call send_status on non-root rank");
  MPIWrapper::send_status(status, comm);
#else
  (void)status;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("broadcast_el", 2) {
  const unsigned int lp = 1;
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  test_comm = mp.get_comm();

  SUBCASE("Throws if not root") {
    if (!mp.is_root()) {
      CHECK_THROWS_WITH_AS(mp.broadcast_el(lp),
                           "Cannot call broadcast_el on non-root rank",
                           std::logic_error);
    }
  }

  if (mp.is_root()) {
    mp.broadcast_el(lp);
  } else {
    const auto res = MPIWrapper::receive_command(test_comm);
    CHECK(res[0] == static_cast<unsigned>(ContextChange::EL));
    CHECK(res[1] == lp);
  }
}
#endif

void MPIObj::broadcast_el(const unsigned lp) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  TIME(ContextChange::EL)
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "Cannot call broadcast_el on non-root rank");
  MPIWrapper::broadcast_el(lp, comm);
#else
  (void)(lp);
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("broadcast_sl", 2) {
  const unsigned int lp = 1;
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  test_comm = mp.get_comm();

  SUBCASE("Throws if not root") {
    if (!mp.is_root()) {
      CHECK_THROWS_WITH_AS(mp.broadcast_sl(lp),
                           "Cannot call broadcast_sl on non-root rank",
                           std::logic_error);
    }
  }

  if (mp.is_root()) {
    mp.broadcast_sl(lp);
  } else {
    const auto res = MPIWrapper::receive_command(test_comm);
    CHECK(res[0] == static_cast<unsigned>(ContextChange::SL));
    CHECK(res[1] == lp);
  }
}
#endif

void MPIObj::broadcast_sl(const unsigned lp,
                          const bool down_sieve) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  TIME((down_sieve) ? ContextChange::SL_REDIST : ContextChange::SL_NO_REDIST)
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "Cannot call broadcast_sl on non-root rank");
  MPIWrapper::broadcast_sl(lp, down_sieve, comm);
#else
  (void)(lp);
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("broadcast_er", 2) {
  const unsigned int rp = 1;
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  test_comm = mp.get_comm();

  SUBCASE("Throws if not root") {
    if (!mp.is_root()) {
      CHECK_THROWS_WITH_AS(mp.broadcast_er(rp),
                           "Cannot call broadcast_er on non-root rank",
                           std::logic_error);
    }
  }

  if (mp.is_root()) {
    mp.broadcast_er(rp);
  } else {
    const auto res = MPIWrapper::receive_command(test_comm);
    CHECK(res[0] == static_cast<unsigned>(ContextChange::ER));
    CHECK(res[1] == rp);
  }
}
#endif

void MPIObj::broadcast_er(const unsigned rp) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  TIME(ContextChange::ER)
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "Cannot call broadcast_er on non-root rank");
  MPIWrapper::broadcast_er(rp, comm);
#else
  (void)(rp);
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("receive_command", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  test_comm = mp.get_comm();

  SUBCASE("Throws if mp is root.") {
    if (mp.is_root()) {
      CHECK_THROWS_WITH_AS(mp.receive_command(),
                           "Cannot call receive_command on root rank",
                           std::logic_error);
    }
  }

  SUBCASE("EL, root") {
    const unsigned lp = 10;
    if (mp.is_root()) {
      mp.broadcast_el(lp);
    } else {
      const auto out = mp.receive_command();
      CHECK(out[0] == static_cast<unsigned>(ContextChange::EL));
      CHECK(out[1] == lp);
    }
  }

  SUBCASE("ER, root") {
    const unsigned lp = 10;
    if (mp.is_root()) {
      mp.broadcast_er(lp);
    } else {
      const auto out = mp.receive_command();
      CHECK(out[0] == static_cast<unsigned>(ContextChange::ER));
      CHECK(out[1] == lp);
    }
  }

  SUBCASE("SL, root") {
    const unsigned lp = 10;
    if (mp.is_root()) {
      mp.broadcast_sl(lp);
    } else {
      const auto out = mp.receive_command();
      CHECK(out[0] == static_cast<unsigned>(ContextChange::SL));
      CHECK(out[1] == lp);
    }
  }
}
#endif

std::array<unsigned, 2> MPIObj::receive_command() const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(is_root(), std::logic_error,
                    "Cannot call receive_command on root rank");
  return MPIWrapper::receive_command(comm);
#else
  // Note; this function won't ever be called in a non-MPI world, so
  // this probably isn't necessary. We could almost certainly turn this
  // into a __builtin_unreachable().
  return std::array<unsigned, 2>();
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("receive_best_lifts_as_root", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};

  SUBCASE("Throws if mp is not root") {
    if (!mp.is_root()) {
      std::vector<LiftEntry> empty;
      CHECK_THROWS_WITH_AS(
          mp.receive_best_lifts_as_root(empty, 0),
          "Cannot call receive_best_lifts_as_root on non-root rank",
          std::logic_error);
    }
  }

  // We test the correctness in mpi_wrapper.hpp
}
#endif

std::vector<LiftEntry> MPIObj::receive_best_lifts_as_root(
    std::vector<LiftEntry> &lifts_in,
    const unsigned full_n) const MPI_DIST_MAY_THROW {

#if defined G6K_MPI
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "Cannot call receive_best_lifts_as_root on non-root rank");
  std::vector<LiftEntry> out;
  MPIWrapper::reduce_best_lifts_to_root(lifts_in, out, full_n, rank,
                                        reduce_best_lifts_op, comm);
  return out;
#else
  (void)full_n;
  return lifts_in;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("share_best_lifts_with_root", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};

  SUBCASE("Cannot be called by root") {
    if (mp.is_root()) {
      std::vector<LiftEntry> empty;
      CHECK_THROWS_WITH_AS(
          mp.share_best_lifts_with_root(empty, 0),
          "Cannot call share_best_lifts_with_root on a root rank",
          std::logic_error);
    }
  }

  // We test the correctness in mpi_wrapper.hpp
}
#endif

void MPIObj::share_best_lifts_with_root(const std::vector<LiftEntry> &lifts_in,
                                        const unsigned full_n) const
    MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(is_root(), std::logic_error,
                    "Cannot call share_best_lifts_with_root on a root rank");
  std::vector<LiftEntry> out;
  MPIWrapper::reduce_best_lifts_to_root(lifts_in, out, full_n, rank,
                                        reduce_best_lifts_op, comm);
  return;
#else
  (void)lifts_in;
  (void)full_n;
  return;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("broadcast_params", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  SieverParams p{};
  // G6K doesn't set this by default.
  p.reserved_n = 100;
  // randomness for the rest is fine.
  p.bdgl_improvement_db_ratio = 1.9;
  p.bgj1_improvement_db_ratio = 1.9;
  p.otf_lift = false;
  p.saturation_ratio = 9;
  // threads is never changed.
  p.threads = 0;

  // seed is sent too
  unsigned int long seed = 5;

  if (mp.is_root()) {
    mp.broadcast_params(p, seed);
  } else {
    auto comm = mp.get_comm();
    SieverParams np;
    uint64_t recv_seed;
    MPIWrapper::receive_params(np, recv_seed, comm);
    CHECK(np.reserved_n == p.reserved_n);
    CHECK(np.bdgl_improvement_db_ratio == p.bdgl_improvement_db_ratio);
    CHECK(np.bgj1_improvement_db_ratio == p.bgj1_improvement_db_ratio);
    CHECK(np.otf_lift == p.otf_lift);
    CHECK(np.saturation_ratio == p.saturation_ratio);
    CHECK(recv_seed == seed);
  }
}
#endif

void MPIObj::broadcast_params(const SieverParams &params,
                              const unsigned int long seed) const
    MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "cannot call broadcast_params with non-root rank");
  MPIWrapper::broadcast_params(params, seed, comm);
#else
  (void)params;
  (void)seed;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("receive_params", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  SieverParams p{};
  // G6K doesn't set this by default.
  p.reserved_n = 100;
  // randomness for the rest is fine.
  p.bdgl_improvement_db_ratio = 1.9;
  p.bgj1_improvement_db_ratio = 1.9;
  p.otf_lift = false;
  p.saturation_ratio = 9;
  // threads is never changed.
  p.threads = 0;

  // The seed is sent too.
  unsigned int long seed = 5;

  if (mp.is_root()) {
    mp.broadcast_params(p, seed);
  } else {
    SieverParams np{};
    unsigned int long recv_seed;
    mp.receive_params(np, recv_seed);
    CHECK(np.reserved_n == p.reserved_n);
    CHECK(np.bdgl_improvement_db_ratio == p.bdgl_improvement_db_ratio);
    CHECK(np.bgj1_improvement_db_ratio == p.bgj1_improvement_db_ratio);
    CHECK(np.otf_lift == p.otf_lift);
    CHECK(np.saturation_ratio == p.saturation_ratio);
    CHECK(seed == recv_seed);
  }
}
#endif

void MPIObj::receive_params(SieverParams &params,
                            uint64_t &seed) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(is_root(), std::logic_error,
                    "cannot call receive_params with a root rank");
  MPIWrapper::receive_params(params, seed, comm);
#else
  (void)params;
  (void)seed;
#endif
}

void MPIObj::receive_initial_setup(SieverParams &p, std::vector<double> &gso,
                                   uint64_t &seed) const MPI_DIST_MAY_THROW {

#if defined G6K_MPI
  THROW_OR_OPTIMISE(is_root(), std::logic_error,
                    "cannot call receive_initial_setup with a root rank");
  MPIWrapper::receive_initial_setup(p, gso, seed, comm);
#else
  (void)p;
  (void)gso;
  (void)seed;
#endif
}

// Note: the tests after this point are to make sure that we can
// actually pass things through the Siever to interact with the rest of the
// world. This is more of a test of the Siever's wrapper rather than anything
// else.

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("can load gso", 2) {
  SieverParams p{};
  p.is_cython = (test_rank == MPIWrapper::global_root_rank);
  p.comm = MPI_Cast::mpi_comm_to_uint64(test_comm);
  p.dist_threshold = 500;

  std::vector<double> arr{1, 2, 3, 4, 5, 6, 7, 8, 9};
  if (test_rank == MPIWrapper::global_root_rank) {
    Siever s{3, arr.data(), p};
  } else {
    MPIObj mp{false, test_comm};
    SieverParams np{};
    std::vector<double> out;
    // Not used.
    unsigned int long recv_seed;

    mp.receive_initial_setup(np, out, recv_seed);
    CHECK(np.dist_threshold == p.dist_threshold);
    CHECK(out == arr);
  }
}
#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("setup", 2) {
  std::vector<double> mu{1, 2, 3, 4, 5, 6, 7, 8, 9};
  SieverParams p{};
  p.reserved_n = 0;
  p.comm = MPI_Cast::mpi_comm_to_uint64(test_comm);
  p.topology = DistSieverType::ShuffleSiever;
  p.is_cython = (test_rank == MPIWrapper::global_root_rank);
  p.simhash_codes_basedir = "test";
  p.sample_by_sums = false;

  if (test_rank == MPIWrapper::global_root_rank) {
    Siever s{3, mu.data(), p};
  } else {
    Siever s{p};

    std::vector<double> mu;
    s.mpi.receive_gso(mu);
    s.load_gso(sqrt(mu.size()), mu.data());

    // Because the root has already broadcasted out, all
    // should be good.
    CHECK(s.full_rr.size() == 3);
    CHECK(s.full_muT.size() == 3);
    CHECK(s.get_params().simhash_codes_basedir == p.simhash_codes_basedir);
    CHECK(s.get_params().sample_by_sums == p.sample_by_sums);

    CHECK(s.mpi.get_topology() == DistSieverType::ShuffleSiever);
    for (unsigned i = 0; i < 3; i++) {
      for (unsigned j = 0; j < 3; j++) {
        if (i == j) {
          CHECK(s.full_muT[i][j] == 1);
          CHECK(s.full_rr[i] == mu[j * 3 + i]);
        } else {
          CHECK(s.full_muT[i][j] == mu[j * 3 + i]);
        }
      }
    }
  }
}
#endif

void MPIObj::reconcile_database(std::vector<CompressedEntry> &cdb,
                                std::vector<Entry> &db,
                                const unsigned n) noexcept {
#if defined G6K_MPI
  if (is_root()) {
    const auto old_size = db.size();
    MPIWrapper::get_database(db, n, comm);
    cdb.resize(db.size());
    for (unsigned i = old_size; i < db.size(); i++) {
      cdb[i].i = i;
    }
  } else {
    // N.B This is a little bit of an abstraction break.
    MPIWrapper::send_database_to_root(db, n, comm);
    db.clear();
    cdb.clear();
  }
  active = false;
#else
  (void)(cdb);
  (void)(db);
  (void)(n);
#endif
}

void MPIObj::setup_database(std::vector<CompressedEntry> &cdb,
                            std::vector<Entry> &db, const unsigned n) noexcept {
#if defined G6K_MPI
  if (is_root()) {
    MPIWrapper::split_database_uids(cdb, db, slot_map, n, comm);
  } else {
    MPIWrapper::receive_database_uids(cdb, db, n, comm);
  }
  active = true;
#else
  (void)(cdb);
  (void)(db);
  (void)(n);
#endif
}

size_t MPIObj::get_cdb_size(const size_t in_db) const noexcept {
#if defined G6K_MPI
  return MPIWrapper::db_size(in_db, comm);
#else
  return in_db;
#endif
}

size_t MPIObj::get_global_saturation(const size_t in_sat) const noexcept {
#if defined G6K_MPI
  if (!active) {
    return in_sat;
  }

  return MPIWrapper::global_saturation(in_sat, comm);
#else
  return in_sat;
#endif
}

size_t MPIObj::db_size(const size_t size) const noexcept {
#if defined G6K_MPI
  if (!active) {
    return size;
  }

  if (is_root()) {
    MPIWrapper::broadcast_db_size(comm);
  }
  return MPIWrapper::db_size(size, comm);
#else
  return size;
#endif
}

size_t MPIObj::db_capacity(const size_t capacity) const noexcept {
#if defined G6K_MPI
  if (!active) {
    return capacity;
  }

  if (is_root()) {
    MPIWrapper::broadcast_db_capacity(comm);
  }
  return MPIWrapper::db_size(capacity, comm);
#else
  return capacity;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("grow_db", 2) {
  std::vector<double> mu{1, 2, 3, 4, 5, 6, 7, 8, 9};
  SieverParams p{};
  p.comm = MPI_Cast::mpi_comm_to_uint64(test_comm);
  p.topology = DistSieverType::ShuffleSiever;
  p.is_cython = (test_rank == MPIWrapper::global_root_rank);

  SUBCASE("Throws on non-root rank") {
    if (test_rank == MPIWrapper::global_root_rank) {
      Siever s{3, mu.data(), p};
    } else {
      Siever s{p};
      CHECK_THROWS_WITH_AS(s.mpi.grow_db(1, 0, s),
                           "Cannot call grow_db with a non-root rank.",
                           std::logic_error);
    }
  }
}
#endif

bool MPIObj::reserve_db(const size_t N,
                        Siever &siever) const MPI_DIST_MAY_THROW {
#ifdef G6K_MPI
  // Prevent recursive instantiations of this function.
  // We also only allow this call to go out if the database is actually split.
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "Cannot call reserve_db with a non-root rank.");

  // Don't do anything in the case that the reservation size is 0: there's
  // no point passing that information on.
  if (N == 0) {
    return false;
  }

  // WARNING: this function acts differently compared to
  // grow_db. In particular, this function always issues a reserve call,
  // even if sieving is not currently distributed.
  // The reason for this is because (for some reason) G6K will call reserve_db
  // with a very large database size at the beginning of certain pumps.
  // This can cause the memory of a local node to be overwhelmed if the
  // requested size is too large: thus, we always pass the request through to
  // distribute the load.
  if (state == MPIObj::State::DEFAULT) {
    TIME(ContextChange::RESERVE);
    state = MPIObj::State::RESIZING;
    // We round up this allocation to include the number of extra vectors we'll
    // need for operation of the BGJ1 distributed sieve. This prevents an extra
    // allocation later on.
    const auto new_size =
        MPIWrapper::reserve_db(N, memory_per_rank, comm) +
        (number_of_ranks() - 1) * (buckets_per_rank) * this->bucket_batches;
    siever.reserve(new_size);
    state = MPIObj::State::DEFAULT;
    return true;
  }
  return false;
#endif
}

bool MPIObj::grow_db(const size_t N, const unsigned large,
                     Siever &siever) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "Cannot call grow_db with a non-root rank.");
  // Prevent recursive instantiation of this function.
  // We also only allow this call to go out if the database is actually split.
  if (state == MPIObj::State::DEFAULT && active) {
    TIME((large) ? ContextChange::GROW_LARGE : ContextChange::GROW_SMALL);
    state = MPIObj::State::RESIZING;
    const auto new_size = MPIWrapper::grow_db(N, large, memory_per_rank, comm);
    siever.grow_db(new_size, large);
    state = MPIObj::State::DEFAULT;
    return true;
  }
  return false;
#else
  (void)N;
  (void)large;
  (void)siever;
  return false;
#endif
}

bool MPIObj::shrink_db(const size_t N,
                       Siever &siever) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  // Prevent recursive calls: the call to shrink_db for a root rank will
  // reinvoke this function, so we have to guard against that here.
  // We also only allow calls to shrink to go through if the database is
  // actually spread out across multiple ranks.
  if (state == MPIObj::State::DEFAULT && active) {
    TIME(ContextChange::SHRINK);
    state = MPIObj::State::RESIZING;
    size_t new_size{};
    // We have to call into MPI if a) `N` is non-zero
    // We have to call into MPI if a) there's something to shrink (indicated by
    // a zero `N`) or b) if we're the root, since we need to inform other ranks
    // regardless.
    if (N != 0 || is_root()) {
      // N.B To keep everything cheap, we need the cdb to be sorted.
      TIME(ContextChange::SORT);
      siever.parallel_sort_cdb();
      new_size = MPIWrapper::shrink_db(N, siever.get_min_length(),
                                       siever.get_max_length(),
                                       siever.get_cdb(), comm);
    }
    siever.shrink_db(new_size);
    state = MPIObj::State::DEFAULT;
    return true;
  }
  return false;
#else
  // Silence warnings.
  (void)N;
  (void)siever;
  return false;
#endif
}

void MPIObj::sort_db() const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "Cannot call sort_db with a non-root rank.");
  // Prevent recursive calls during shrinking: each rank will already sort their
  // cdbs during this portion.
  if (state != MPIObj::State::RESIZING) {
    TIME(ContextChange::SORT);
    MPIWrapper::sort_db(comm);
  }
#else
  return;
#endif
}

void MPIObj::gso_update_postprocessing(const unsigned int l_,
                                       const unsigned int r_,
                                       const unsigned old_n,
                                       const bool should_redist,
                                       long const *M) MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(
      !is_root(), std::logic_error,
      "Cannot call gso_update_postprocessing on a non-root rank.");
  TIME(ContextChange::GSO_PP);
  MPIWrapper::gso_update_postprocessing(l_, r_, old_n, M, should_redist, comm);
#else
  (void)l_;
  (void)r_;
  (void)old_n;
  (void)M;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("send_stop", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};

  if (mp.is_root()) {
    mp.send_stop();
  } else {
    const auto status = mp.receive_command();
    CHECK(status[0] == ContextChange::STOP);
    // The second parameter here doesn't matter.
  }
}
#endif
void MPIObj::send_stop() MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "Cannot call send_stop on a non-root rank.");
  TIME(ContextChange::STOP);
  MPIWrapper::send_stop(comm);
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("receive_gso_update_postprocessing", 2) {
  constexpr unsigned l_ = 49;
  constexpr unsigned r_ = 120;

  constexpr unsigned n = r_ - l_;
  // Size before context change.
  constexpr unsigned old_n = n - 1;

  std::array<long, n * old_n> M;
  std::iota(M.begin(), M.end(), 0);

  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};

  std::iota(M.begin(), M.end(), 0);
  if (mp.is_root()) {
    auto comm = mp.get_comm();
    MPIWrapper::gso_update_postprocessing(l_, r_, old_n, M.data(), comm);
  } else {
    const auto header = mp.receive_command();
    CHECK(header[0] == ContextChange::GSO_PP);
    CHECK(header[1] == n);
    std::vector<long> M_new;
    const auto recv = mp.receive_gso_update_postprocessing(M_new, old_n);
    CHECK(recv[0] == l_);
    CHECK(recv[1] == r_);
    CHECK(M_new.size() == n * old_n);
    CHECK(memcmp(M_new.data(), M.data(), sizeof(long) * (n * old_n)) == 0);
  }
}
#endif

std::array<unsigned, 3> MPIObj::receive_gso_update_postprocessing(
    std::vector<long> &M, const unsigned old_n) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  TIME(ContextChange::GSO_PP);
  THROW_OR_OPTIMISE(
      is_root(), std::logic_error,
      "Cannot call receive_gso_update_postprocessing on a root rank");
  return MPIWrapper::receive_gso_update_postprocessing(M, old_n, comm);
#else
  (void)M;
  (void)old_n;
  return std::array<unsigned, 3>();
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("start_bgj1", 2) {
  constexpr double alpha = 5.0;
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  if (mp.is_root()) {
    mp.start_bgj1(alpha);
  } else {
    const auto status = mp.receive_command();
    CHECK(status[0] == ContextChange::BGJ1);
    double alpha_in;
    MPI_Bcast(&alpha_in, 1, MPI_DOUBLE, MPIWrapper::global_root_rank,
              mp.get_comm());
    CHECK(alpha_in == alpha);
  }
}
#endif

void MPIObj::start_bgj1(const double alpha) const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  TIME(ContextChange::BGJ1);
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "Cannot call start_bgj1 on a non-root rank");
  MPIWrapper::start_bgj1(alpha, comm);
#else
  (void)alpha;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("receive_alpha", 2) {
  constexpr double alpha = 5.0;
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  if (mp.is_root()) {
    mp.start_bgj1(alpha);
  } else {
    const auto status = mp.receive_command();
    CHECK(status[0] == ContextChange::BGJ1);
    CHECK(mp.receive_alpha() == alpha);
  }
}
#endif

double MPIObj::receive_alpha() const MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(is_root(), std::logic_error,
                    "Cannot call receive_alpha on a root rank");
  return MPIWrapper::receive_alpha(comm);
#else
  return 0.0;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("is_in_context_change", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};
  // We start not in a context change.
  CHECK(!mp.is_in_context_change());
  // Mark as being in one.
  mp.in_context_change();
  CHECK(mp.is_in_context_change());
  // Mark as being out of one.
  mp.out_of_context_change();
  CHECK(!mp.is_in_context_change());
}
#endif

void MPIObj::broadcast_initialize_local(const unsigned ll, const unsigned l,
                                        const unsigned r) MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(
      !is_root(), std::logic_error,
      "Cannot call broadcast_initialize_local on a non-root rank");
  // We don't allow ILs inside context change operations because it implies
  // repeated work.
  if (state != State::CONTEXT_CHANGE) {
    TIME(ContextChange::IL);
    MPIWrapper::broadcast_initialize_local(ll, l, r, comm);
  }
#else
  (void)ll;
  (void)l;
  (void)r;
#endif
}

std::array<unsigned, 2> MPIObj::receive_l_and_r() MPI_DIST_MAY_THROW {
#if defined G6K_MPI
  THROW_OR_OPTIMISE(is_root(), std::logic_error,
                    "Cannot call receive_l_and_r on a root rank");
  return MPIWrapper::receive_l_and_r(comm);
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("broadcast_il", 2) {
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};

  constexpr unsigned ll = 1;
  constexpr unsigned l = 2;
  constexpr unsigned r = 3;

  if (mp.is_root()) {
    mp.broadcast_initialize_local(ll, l, r);
  } else {
    const auto status = mp.receive_command();
    CHECK(status[0] == ContextChange::IL);
    CHECK(status[1] == ll);

    // Must receive l and r separately.
    const auto l_and_r = mp.receive_l_and_r();
    CHECK(l_and_r[0] == l);
    CHECK(l_and_r[1] == r);
  }
}
#endif

void MPIObj::build_global_histo(long *const histo,
                                const bool need_message) noexcept {
#ifdef G6K_MPI
  assert(histo);
  TIME(ContextChange::HISTO);
  if (is_root() && need_message) {
    MPIWrapper::broadcast_build_histo(comm);
  }
  MPIWrapper::build_global_histo(histo, comm);
#else
  (void)histo;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("build_global_histo", 2) {
  std::array<long, Siever::size_of_histo> a1, a2, expected;
  std::iota(a1.begin(), a1.end(), 0);
  std::iota(a2.begin(), a2.end(), 0);
  for (unsigned i = 0; i < a1.size(); i++) {
    expected[i] = a1[i] + a2[i];
  }

  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm};

  if (mp.is_root()) {
    mp.build_global_histo(a1.data());
    CHECK(a1 == expected);
  } else {
    const auto status = mp.receive_command();
    CHECK(status[0] == ContextChange::HISTO);
    mp.build_global_histo(a2.data());
  }
}
#endif

double MPIObj::gather_gbl_sat_variance(const size_t cur_sat) noexcept {
#ifdef G6K_MPI
  return MPIWrapper::gather_gbl_sat_variance(cur_sat, comm);
#else
  (void)cur_sat;
  return 0.0;
#endif
}

double MPIObj::gather_gbl_ml_variance(const double cur_gbl_ml) noexcept {
#ifdef G6K_MPI
  return MPIWrapper::gather_gbl_ml_variance(cur_gbl_ml, comm);
#else
  (void)cur_gbl_ml;
  return 0.0;
#endif
}

void MPIObj::setup_bucket_positions(const size_t size) noexcept {
#ifdef G6K_MPI
  bgj1_buckets.setup_bucket_positions(size, bucket_batches, buckets_per_rank,
                                      number_of_ranks(), rank);
#else
  (void)size;
#endif
}

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("setup_bucket_positions", 3) {
  static constexpr std::array<uint64_t, 3> threads{30, 90, 15};
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm, 500,
            threads[test_rank]};

  mp.setup_bucket_positions(5);

  // The first entry for each rank is 5.
  MPI_CHECK(0, mp.get_bucket_position(1) == 5);
  MPI_CHECK(1, mp.get_bucket_position(0) == 5);
  MPI_CHECK(2, mp.get_bucket_position(0) == 5);

  // The second entry for each rank is the number of threads
  // held by the first selected rank, plus the size.
  MPI_CHECK(0, mp.get_bucket_position(2) == 95);
  MPI_CHECK(1, mp.get_bucket_position(2) == 35);
  MPI_CHECK(2, mp.get_bucket_position(1) == 35);
}
#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("setup_size_requests", 3) {
  static constexpr std::array<uint64_t, 3> threads{30, 90, 15};
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm, 500,
            threads[test_rank]};

  SUBCASE("bucket sizes") {
    // It's also useful to know that we aren't reading into invalid memory.
    const auto sizes = mp.get_bucket_sizes();
    CHECK(sizes.size() == 2);
    for (const auto &size : sizes) {
      // We read in this many centers per go.
      CHECK(size.size() == threads[test_rank]);
    }
  }

  SUBCASE("size_requests") {
    auto requests = mp.get_size_requests();
    CHECK(requests.size() == 2);
    for (auto &req : requests) {
      // N.B non-started requests are marked as being finished in MPI.
      int flag;
      MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
      CHECK(flag);
    }
  }
}
#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("setup_outgoing_size_requests", 3) {
  static constexpr std::array<uint64_t, 3> threads{30, 90, 15};
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm, 500,
            threads[test_rank]};

  SUBCASE("outgoing_bucket_sizes") {
    const auto sizes = mp.get_outgoing_bucket_sizes();
    // All but us
    REQUIRE(sizes.size() == 2);
    // Check each rank has allocated enough storage for the other ranks.
    MPI_CHECK(0, sizes[0].size() == threads[1]);
    MPI_CHECK(0, sizes[1].size() == threads[2]);
    MPI_CHECK(1, sizes[0].size() == threads[0]);
    MPI_CHECK(1, sizes[1].size() == threads[2]);
    MPI_CHECK(2, sizes[0].size() == threads[0]);
    MPI_CHECK(2, sizes[1].size() == threads[1]);
  }

  SUBCASE("outgoing_size_requests") {
    auto requests = mp.get_outgoing_size_requests();
    CHECK(requests.size() == 2);
    for (auto &req : requests) {
      // N.B non-started requests are marked as finished.
      int flag;
      MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
      CHECK(flag);
    }
  }
}
#endif

#if defined G6K_MPI && !defined DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("start_size_requests", 2) {
  // This test essentially just checks that the requests we create line up.
  static constexpr std::array<uint64_t, 2> threads{30, 90};
  MPIObj mp{(test_rank == MPIWrapper::global_root_rank), test_comm, 500,
            threads[test_rank]};

  // We'll send a sequence to each rank.
  std::vector<std::vector<uint64_t>> out_arrs(2);
  out_arrs[0].resize(threads[0]);
  out_arrs[1].resize(threads[1]);
  std::iota(out_arrs[0].begin(), out_arrs[0].end(), 0);
  std::iota(out_arrs[1].begin(), out_arrs[1].end(), 0);

  // Now we write the sequences to the pre-allocated storage.
  auto &outgoing_sizes = mp.get_outgoing_bucket_sizes();
  CHECK(outgoing_sizes.size() == 1);
  MPI_CHECK(0, outgoing_sizes[0].size() == threads[1]);
  MPI_CHECK(1, outgoing_sizes[0].size() == threads[0]);

  const auto index = (test_rank == 0) ? 1 : 0;
  std::copy(out_arrs[index].cbegin(), out_arrs[index].cend(),
            outgoing_sizes[0].begin());

  // This indicates the background requests should start. MPI reads from the
  // outgoing_sizes buffer if we've set everything up properly.
  auto outgoing_size_requests = mp.get_outgoing_size_requests();
  MPI_Startall(outgoing_size_requests.size(), outgoing_size_requests.data());
  mp.start_size_requests();

  // Wait for the requests to go through.
  auto &size_requests = mp.get_size_requests();
  MPI_Waitall(size_requests.size(), size_requests.data(), MPI_STATUSES_IGNORE);

  MPI_CHECK(0, mp.get_bucket_sizes()[0] == out_arrs[0]);
  MPI_CHECK(1, mp.get_bucket_sizes()[0] == out_arrs[1]);
}
#endif

void MPIObj::init_thread_entries_bgj1(const unsigned nr_threads) noexcept {
#ifdef G6K_MPI
  thread_entries.resize(nr_threads);
  for (auto &v : thread_entries) {
    std::vector<Entry> tmp;
    v.bucket.swap(tmp);
  }

  insertion_dbs.resize(nr_threads);
  for (auto &v : insertion_dbs) {
    v.resize(number_of_ranks());
  }
#endif
}

void MPIObj::init_thread_entries_bdgl(const unsigned nr_threads) noexcept {
#ifdef G6K_MPI
  bdgl_thread_entries = std::vector<BdglThreadEntry>(nr_threads);
#endif
}

void MPIObj::initialise_thread_entries(const unsigned nr_threads,
                                       const bool is_bdgl) noexcept {
#ifdef G6K_MPI

  if (!is_bdgl) {
    init_thread_entries_bgj1(nr_threads);
  } else {
    init_thread_entries_bdgl(nr_threads);
  }

  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

void MPIObj::collect_statistics(SieveStatistics &stats) noexcept {
#ifdef G6K_MPI
  if (is_root()) {
    MPIWrapper::deserialise_stats_and_add(stats, comm);
  } else {
    MPIWrapper::serialise_stats(stats, comm);
  }
#else
  (void)stats;
  return;
#endif
}

void MPIObj::finish_bdgl() noexcept {
#ifdef G6K_MPI
  stopped = false;
  issued_stop = false;
  is_barrier_empty = true;
  is_used = true;
  requests.clear_reqs();

  MPI_Wait(requests.stop_req(), MPI_STATUS_IGNORE);
  std::fill(scratch_lookup.begin(), scratch_lookup.end(), 0);
  free_scratch.store(scratch_buffers, std::memory_order_relaxed);
  wait_head = 0;
#endif
}

void MPIObj::cancel_outstanding() noexcept {
#ifdef G6K_MPI
  // This function is surprisingly tricky to get right in a fully async sieve.
  // To make this easier to handle, we enforce at the calling level that all
  // requests have been processed, so here we just have to tidy up some.
  // Put simply: we just reset all of the various invariants.

  // This portion is essentially resetting all of the invariants for
  // the next sieving loop. We have to do this to make sure
  // everything starts properly next time.
  bgj1_buckets.reset_comms();

  bgj1_buckets.reset_counts();
  bgj1_buckets.reset_ops();

  stopped = false;
  issued_stop = false;
  is_barrier_empty = true;
  is_used = true;

  sat_drop.store(0);
  requests.clear_reqs();

  MPI_Type_free(&vector_type);
  MPI_Type_free(&entry_type);

  MPI_Wait(requests.stop_req(), MPI_STATUS_IGNORE);

  // Finally, we'll shrink each of the outgoing insertion vectors.
  // This is useful to prevent the memory usage from getting out of control.
  for (auto &v : tmp_insertion_buf) {
    for (auto &y : v.buffers) {
      y.buffer.clear();
      y.buffer.shrink_to_fit();
    }
  }

  for (auto &v : insertion_vector_bufs) {
    v.outgoing_buffer.clear();
    v.incoming_buffer.clear();
    v.outgoing_buffer.shrink_to_fit();
    v.incoming_buffer.shrink_to_fit();
  }

  // And the individual ones.
  for (auto &set : insertion_dbs) {
    for (auto &v : set) {
      v.clear();
      v.shrink_to_fit();
    }
  }

  for (auto &vector : scratch_space) {
    decltype(vector.incoming_scratch) tmp;
    vector.incoming_scratch.swap(tmp);
    decltype(vector.outgoing_scratch) tmp2;
    vector.outgoing_scratch.swap(tmp2);
  }

  for (auto &v : scratch_used) {
    v.store(0, std::memory_order_relaxed);
  }

  std::fill(scratch_lookup.begin(), scratch_lookup.end(), 0);
  free_scratch.store(scratch_buffers, std::memory_order_relaxed);
  wait_head = 0;
  std::fill(finished_sizes.begin(), finished_sizes.end(), 0);
#else
  return;
#endif
}

void MPIObj::reset_stats() const noexcept {
#ifdef G6K_MPI
  TIME(ContextChange::RESET_STATS);
  THROW_OR_OPTIMISE(!is_root(), std::logic_error,
                    "cannot call reset_stats on a non-root rank");
  MPIWrapper::reset_stats(comm);
#else
  return;
#endif
}

double MPIObj::get_min_max_len(double max_len) const noexcept {
#ifdef G6K_MPI
  return MPIWrapper::get_min_max_len(max_len, comm);
#else
  __builtin_unreachable();
#endif
}

void MPIObj::redistribute_database(std::vector<CompressedEntry> &cdb,
                                   std::vector<Entry> &db, const unsigned n,
                                   UidHashTable &hash_table,
                                   std::vector<unsigned> &duplicates,
                                   std::vector<unsigned> &incoming) {
#ifdef G6K_MPI
  TIME(ContextChange::REDIST_DB);
  return MPIWrapper::redistribute_database(db, cdb, slot_map, n, hash_table,
                                           comm, duplicates, incoming);
#else
  __builtin_unreachable();
#endif
}

void MPIObj::add_to_outgoing(const Entry &e, const size_t index,
                             const unsigned n) noexcept {
#ifdef G6K_MPI
  if (e.uid == 0)
    return;

  // Work out where in the array it is going to live.
  const auto owner = slot_map[UidHashTable::get_slot(e.uid)];
  // Make sure that we don't own it. This shouldn't happen: in fact, it should
  // be guarded by the caller, so we'll only do this in debug modes.
  assert(!owns_uid(e.uid));

  auto &buf = tmp_insertion_buf[index].buffers[owner];
  std::lock_guard<std::mutex> lock(buf.lock);
  buf.buffer.insert(buf.buffer.end(), e.x.cbegin(), e.x.cbegin() + n);
#else
  __builtin_unreachable();
#endif
}

unsigned MPIObj::add_to_outgoing_bdgl(const Entry &entry, const unsigned index,
                                      const unsigned n) noexcept {
#ifdef G6K_MPI
  // In BDGL the insertion code is far more straightforward: we grow a single,
  // large buffer that is keyed into by various other insertion guards etc.
  // Because we insert locally, all we really need to do is return the position
  // that we inserted at.

  // NOTE: We've already locked this structure earlier, so inserting without it
  // is fine.
  auto &t_entry = bdgl_thread_entries[index];
  auto &buffer = t_entry.insert_db;
  const auto pos = buffer.size();
  buffer.resize(pos + n);
  std::copy(entry.x.cbegin(), entry.x.cbegin() + n, buffer.begin() + pos);
  return pos;
#endif
}

void MPIObj::batch_insertions(const size_t index, const unsigned batch,
                              const unsigned n) noexcept {
#ifdef G6K_MPI
  /*
  // This function simply inserts all finished insertion databases into the
  // set of global insertions.
  auto &insertion_set = insertion_dbs[index];
  const auto size = insertion_set.size();
  auto &insertion_buffers = tmp_insertion_buf[batch];

  for (unsigned i = 0; i < size; i++) {
    if (insertion_set[i].empty())
      continue;
    auto &insertion_entry = insertion_buffers.buffers[i];
    std::lock_guard<std::mutex> lock(insertion_entry.lock);
    insertion_entry.buffer.insert(insertion_entry.buffer.end(),
                                  insertion_set[i].cbegin(),
                                  insertion_set[i].cend());
  }

  for (auto &v : insertion_set) {
    v.clear();
  }
  */
#endif
}

void MPIObj::setup_insertion_requests(const unsigned n,
                                      const unsigned lift_bound_size) noexcept {
#ifdef G6K_MPI
  vector_type = Layouts::get_entry_vector_layout(n);
  entry_type = Layouts::get_entry_type(n);
  this->size_for_header = size_for_headers(lift_bound_size);
  bgj1_buckets.setup_sync_objects(number_of_ranks() * this->size_for_header);

#ifdef MPI_TRACK_UNIQUES
  unique_sends = 0;
  nr_sends = 0;
#endif

  // Not clear we need to do anything else here: serialisation is likely fine
  // "as is".

#else
  return;
#endif
}

void MPIObj::start_batch(const std::vector<Entry> &db,
                         const std::vector<CompressedEntry> &centers,
                         const unsigned batch, const int64_t trial_count,
                         const std::vector<FT> &lift_bounds) noexcept {
#ifdef G6K_MPI
  auto &scratch_space = bgj1_buckets.get_memory_for(batch);
  SyncHeader header{trial_count < 0, sat_drop.exchange(0)};
  // N.B Because this is an Allgather we just place everything in the
  // right portion of the scratch space and then use MPI_IN_PLACE.
  // The "right position" is the size needed * our rank.
  auto position = size_for_header * rank;
  auto &communicator = bgj1_buckets.get_comm(batch).get_comm();
  MPI_Pack(&header, 1, sync_header_type, scratch_space.data(),
           scratch_space.size(), &position, communicator);
  MPI_Pack(lift_bounds.data(), lift_bounds.size(), Layouts::get_data_type<FT>(),
           scratch_space.data(), scratch_space.size(), &position, communicator);

  // Make the type once to make packing faster.
  auto segment_type = Layouts::get_entry_layout_non_contiguous(
      centers.cbegin(), centers.cend(), vector_type);

  MPI_Pack(db.data(), 1, segment_type, scratch_space.data(),
           scratch_space.size(), &position, communicator);

  // N.B For MPI_PACKED messages the sendcount and recvcount parameters take on
  // a new meaning, denoting the number of bytes that are being sent and
  // received respectively from each process. Thus, we set this to header_size.
  MPI_Iallgather(MPI_IN_PLACE, size_for_header, MPI_PACKED,
                 scratch_space.data(), size_for_header, MPI_PACKED,
                 communicator, requests.bucketing_requests(batch));

  TRACK(ContextChange::BGJ1_CENTER_SEND, size_for_header);
  MPI_Type_free(&segment_type);
  bgj1_buckets.start_incoming_centers(batch);

#else
  return;
#endif
}

void MPIObj::grab_lift_bounds(std::vector<FT> &lift_bounds,
                              FT &lift_max_bound) noexcept {
#ifdef G6K_MPI
  MPIWrapper::grab_lift_bounds(lift_bounds, lift_max_bound, comm);
#else
  return;
#endif
}

void MPIObj::test() noexcept {
#ifdef G6K_MPI
  // This carries out the underlying MPI message exchange.
  requests.test(rank, is_barrier_empty);

  // We need to check if we've stopped.
  stopped = requests.nr_finished_stop_req();

  // This is here because we need to be able to potentially retain finished size
  // requests across many iterations.
  for (const auto size : requests.finished_size_reqs()) {
    finished_sizes[size] = true;
  }

#endif
}

void MPIObj::process_stop() noexcept {
#ifdef G6K_MPI
  // We need to ensure that we are in a synchronous state. Essentially,
  // because of the collective communication rules we have to ensure that all
  // processes are in the same state when the sieving iteration exits. Sadly,
  // MPI does not support explicitly cancelling collective requests, see
  // https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf, page 251, line 3.
  // It turns out the best way to do this is to simply count the number of reads
  // / writes issued for centers across all ranks and then roll forward until we
  // hit the maximum across all ranks. Otherwise, we end up with a plethora of
  // race conditions (MPI tests are local operations, which means that process
  // `i` can report an operation as finished before process `j` has even started
  // it).
  MPI_Wait(requests.stop_req(), MPI_STATUS_IGNORE);
  auto &counts = bgj1_buckets.counts();
  auto &states = bgj1_buckets.states();
  // N.B Have to strip off the reference, or value_type is not defined.
  using T = std::remove_reference_t<decltype(counts)>::value_type;
  if (rank == MPIWrapper::global_root_rank) {
    TRACK(ContextChange::BGJ1_PROCESS_STOP, sizeof(T) * counts.size());
  }
  MPI_Allreduce(counts.data(), states.data(), counts.size(),
                Layouts::get_data_type<T>(), MPI_MAX, stop_comm);
  // And finally, we need to make it so we don't have to do all of this again.
  stopped = false;
  issued_stop = true;
  is_barrier_empty = true;

  // We do not adjust the wait_head here: we will deal with it implicitly in
  // what follows. The logic for this is in forward_and_gather_all, but
  // essentially, we continue using the implicit barrier of wait_head (along
  // with the newly gathered count information) to ensure that we complete all
  // of the outstanding requests in order. Once they're all done, we quit.
#endif
}

int MPIObj::get_root_rank() const noexcept {
#if defined G6K_MPI
  return MPIWrapper::global_root_rank;
#else
  return 0;
#endif
}

int MPIObj::size_for_headers(const unsigned size) noexcept {
#ifdef G6K_MPI
  // Work out how much size we need for each piece of scratch space.
  int size_for_header, size_for_syncs, size_for_vectors;
  MPI_Pack_size(1, sync_header_type, comm, &size_for_header);
  MPI_Pack_size(size, Layouts::get_data_type<FT>(), comm, &size_for_syncs);
  MPI_Pack_size(buckets_per_rank, vector_type, comm, &size_for_vectors);
  return size_for_header + size_for_syncs + size_for_vectors;
#else
  return 0;
#endif
}

MPIObj::SyncChange MPIObj::process_incoming_syncs(std::vector<FT> &lift_bounds,
                                                  std::vector<Entry> &db,
                                                  FT &lift_max_bound) noexcept {
#ifdef G6K_MPI
  SyncChange sync_change{0, 0};

  const auto nr_ranks = static_cast<unsigned>(number_of_ranks());
  const auto size = lift_bounds.size();
  assert(rank >= 0);

  // Finished syncs are done in terms of particular incoming set of buckets.
  const auto finished_reqs = requests.finished_bucketing_reqs();

  // We unpack the sync headers directly into this variable.
  SyncHeader header;

  // N.B This technically wastes some space, but
  // it's likely faster than doing a heap allocation on each
  // iteration.
  std::array<FT, MAX_SIEVING_DIM> lift_tmp;

  for (const auto batch : finished_reqs) {
    // Each other rank will have sent sync information over.
    // These are packed into the receive buffer, which contains a sync header,
    // the lift bounds and the vectors that are to be inserted into the
    // database. We deal with this by iterating and unpacking accordingly.
    const auto &buffer = bgj1_buckets.get_memory_for(batch);
    const auto buffer_size = buffer.size();
    const auto buff = buffer.data();
    int position{};

    for (unsigned i = 0; i < nr_ranks; i++) {
      // NOTE: if i == rank we need to increment the `position` value by
      // the size of the message. This is because an Allgather places our
      // outgoing elements in the buffer, which would cause us to miss a
      // different parties' values.
      // NOTE NOTE: however, we still unpack the header. Why? Because it has the
      // stopping information in it, and that might be useful for us (it's
      // actually better than checking on each iteration of the calling sieve
      // because of memory ordering constraints).
      if (i == static_cast<unsigned>(rank)) {
        // We use a temporary for this because it makes the maths easier.
        auto old_pos = position;
        MPI_Unpack(buff, buffer_size, &old_pos, &header, 1, sync_header_type,
                   comm);
        sync_change.any_sub_zero |= header.trial_count;
        position += size_for_header;
        continue;
      }

      // And finally we unpack the rest of the stuff. This follows the same
      // serialisation layout as in start_batch.
      MPI_Unpack(buff, buffer_size, &position, &header, 1, sync_header_type,
                 comm);
      sync_change.any_sub_zero |= header.trial_count;
      sync_change.sat_drop += header.sat_drop;

      MPI_Unpack(buff, buffer_size, &position, lift_tmp.data(), size,
                 Layouts::get_data_type<FT>(), comm);
      for (unsigned j = 0; j < size; j++) {
        lift_bounds[j] = std::min(lift_bounds[j], lift_tmp[j]);
      }

      // Unpack the entries too.
      MPI_Unpack(buff, buffer_size, &position,
                 db.data() +
                     bgj1_buckets.bucket_position(i, batch, bucket_batches),
                 buckets_per_rank, vector_type, comm);
    }
  }

  // If there were actually some incoming syncs, update the maximum bound.
  if (finished_reqs.size > 0) {
    lift_max_bound =
        *std::max_element(lift_bounds.cbegin(), lift_bounds.cend());
  }

  return sync_change;
#endif
}

FT MPIObj::get_norm(const size_t N, const Siever &siever) noexcept {
#ifdef G6K_MPI
  return MPIWrapper::get_norm(N, siever.get_min_length(),
                              siever.get_max_length(), siever.get_cdb(),
                              ContextChange::SHRINK, comm);
#else
  return 0.0;
#endif
}

FT MPIObj::get_norm_bdgl(const size_t N, const Siever &siever) noexcept {
#ifdef G6K_MPI
  return MPIWrapper::get_norm(N, siever.get_min_length(),
                              siever.get_max_length(), siever.get_cdb(),
                              ContextChange::BDGL, comm);
#else
  return 0.0;
#endif
}

void MPIObj::send_sizes(
    const unsigned batch,
    const std::vector<std::vector<std::vector<unsigned>>> &cbuckets) noexcept {
#ifdef G6K_MPI

  // Load all of the finished buckets. This acquire buffer will catch all of the
  // writes that have been done by the incremented atomic counts, which means
  // that this should catch all flushes etc.
  std::atomic_thread_fence(std::memory_order_acquire);

  auto &sizes = bgj1_buckets.get_sizes(batch);
  std::fill(sizes.begin(), sizes.end(), 0);
  const auto nr_ranks = static_cast<unsigned>(number_of_ranks());
  // sizes is indexed by rank: we expect each rank to have buckets_per_rank
  // entries. cbuckets is also indexed by rank.
  for (unsigned i = 0; i < nr_ranks; i++) {
    if (i == unsigned(rank))
      continue;
    const unsigned size = cbuckets[i].size();

    for (unsigned j = 0; j < size; j++) {
      sizes[i * buckets_per_rank + j] = cbuckets[i][j].size();
    }
  }

  auto &communicator = bgj1_buckets.get_comm(batch);
  using T = std::remove_reference_t<decltype(sizes)>::value_type;

  TRACK(ContextChange::BGJ1_SEND_SIZE,
        sizeof(T) * (sizes.size() - buckets_per_rank));

  MPI_Ialltoall(MPI_IN_PLACE, buckets_per_rank, MPI_INT, sizes.data(),
                buckets_per_rank, MPI_INT, communicator.get_comm(),
                requests.size_requests(batch));
  bgj1_buckets.start_incoming_size(batch);
#endif
}

void MPIObj::forward_and_gather_all(
    const unsigned n,
    const std::vector<std::vector<std::vector<std::vector<unsigned>>>>
        &cbuckets,
    const std::vector<Entry> &db) noexcept {
#ifdef G6K_MPI
  // N.B As we are the only ones responsible for decrementing this, we
  // just copy it.
  const auto nr_to_use = free_scratch.load(std::memory_order_relaxed);
  if (nr_to_use == 0) {
    return;
  }

  // If we've already issued a stop then we need to be careful here: it's very
  // possible that we are in some out-of-order situation. Thus, we need to enter
  // this block in all "issued_stop" settings.

  // The way this code works is as follows. We use the implicit synchronisation
  // that is already provided by wait_head to establish a somewhat sequential
  // order for sends across all ranks.
  //
  // We consider a batch for use as the wait head in the following settings:
  // 1. If the batch is being sieved and there's more batches to send from us
  //    (this happens if the counts don't match).
  // 2. If the batch is being used for bucketing or receiving sizes, but the
  // counts
  //    match. This can only be true: there's never a situation in which the
  //    counts can't match here, because that would imply that the batches are
  //    out of sync across ranks.
  // 3. If the batch is currently receiving centres. In this world, the counts
  // must
  //    also match, and so this is a safe choice.
  //
  // We follow this in an incrementing order (similar to the original
  // synchronisation mechanism from below), since this gives us a loose barrier
  // across all of the ranks.
  //
  // Finally, if there's no such batch available (for whatever reason) then we
  // are done with sieving the batches, and thus we just mark the wait_head as
  // unused.
  if (issued_stop) {
    if (bgj1_buckets.is_finished(wait_head)) {
      const auto pos = bgj1_buckets.get_next_wait_head(wait_head);
      if (pos == bucket_batches) {
        is_used = false;
        return;
      }
      wait_head = pos;
    }

    if (finished_sizes[wait_head] && is_used &&
        bgj1_buckets.is_sizing(wait_head)) {
      std::atomic_thread_fence(std::memory_order_acquire);
      unsigned used{};

      do {
        forward_and_gather_buckets(n, wait_head, cbuckets[wait_head], db);
        finished_sizes[wait_head] = false;
        ++used;

        // Find the first active size request in the bucket set, or the
        // current outgoing bucketing request.
        const auto pos = bgj1_buckets.get_next_wait_head(wait_head);

        // If there aren't any, then we are done, so just exit.
        if (pos == bucket_batches) {
          is_used = false;
          break;
        }

        // Otherwise update the wait_head.
        wait_head = pos;
        // The wait head can be updated for many reasons: it doesn't always
        // indicate that we're done with a size request. We check to make sure.
        if (!bgj1_buckets.is_sizing(wait_head)) {
          break;
        }
      } while (used < nr_to_use && finished_sizes[wait_head]);

      free_scratch.fetch_sub(used, std::memory_order_relaxed);
    }
    return;
  }

  // In all other circumstances, we just need to do an approximate forward
  // barrier. Check if the singular one that we're waiting on is finished.
  if (finished_sizes[wait_head] && bgj1_buckets.is_sizing(wait_head)) {
    // Acquire all of the writes to the buckets that we're sending.
    std::atomic_thread_fence(std::memory_order_acquire);
    unsigned used{};
    do {
      forward_and_gather_buckets(n, wait_head, cbuckets[wait_head], db);
      ++used;
      finished_sizes[wait_head] = false;
      wait_head = (wait_head == bucket_batches - 1) ? 0 : wait_head + 1;
      if (!bgj1_buckets.is_sizing(wait_head)) {
        break;
      }
    } while (used < nr_to_use && finished_sizes[wait_head]);

    free_scratch.fetch_sub(used, std::memory_order_relaxed);
  }

#endif
}

unsigned MPIObj::get_unused_scratch_buffer(const int batch) noexcept {
#ifdef G6K_MPI
  const auto index = [this, batch]() {
    if (bucket_batches == scratch_buffers) {
      return static_cast<unsigned>(batch);
    }

    // Find the first unused scratch vector.
    auto ins_iterator =
        std::find_if(scratch_used.begin(), scratch_used.end(),
                     [](const auto &val) { return val.load() == 0; });
    assert(ins_iterator != scratch_used.end());
    return static_cast<unsigned>(
        std::distance(scratch_used.begin(), ins_iterator));
  }();

  // Mark the batch as used.
  // N.B This is non-atomic.
  scratch_lookup[batch] = index;
  return index;
#endif
}

template <bool is_bgj1>
void MPIObj::dispatch_to_forward_and_gather_buckets(
    const unsigned n, const int batch,
    const std::vector<std::vector<std::vector<unsigned>>> &cbuckets,
    const std::vector<Entry> &db, const std::vector<int> &sizes,
    MPI_Comm communicator) noexcept {
#ifdef G6K_MPI

  // N.B In BDGL land we have to use the scratch_vector before this point (for
  // various storage reasons). For BGJ1 it's slightly different, so we'll handle
  // it separately.
  const auto index =
      (is_bgj1) ? get_unused_scratch_buffer(batch) : scratch_lookup[batch];
  auto &scratch_vector = scratch_space[index];
  auto req = requests.bucketing_replies(batch);
  auto &bucket_pairs = get_bucket_pairs(batch);
  MPIWrapper::forward_and_gather_buckets(
      n, buckets_per_rank, cbuckets, db, sizes, scratch_vector.incoming_scratch,
      scratch_vector.outgoing_scratch, bucket_pairs, scounts, sdispls, rcounts,
      rdispls, entry_type, communicator, req);

  if (is_bgj1) {
    bgj1_buckets.start_incoming_buckets(batch);
  }

  // N.B This probably doesn't actually need to be a release barrier, but this
  // is just for safety (it makes sure that the e.g bucket_pairs are correctly
  // lined up).
  if (is_bgj1) {
    scratch_used[index].store(memory_per_rank[rank], std::memory_order_release);
  } else {
    // For BDGL we need to do this slightly differently, because we might have a
    // different number of buckets each time this runs.
    // N.B The "max" here is to handle the case that we've got a batch size of
    // 0: we still need to be able to mark this particular buffer as freed
    // later.
    const auto to_add = std::max<unsigned>(bdgl_batch_info[batch].size, 1);
    bdgl_bucket_active[batch].store(to_add, std::memory_order_relaxed);
    scratch_used[index].store(to_add, std::memory_order_release);
  }
#endif
}

void MPIObj::forward_and_gather_buckets(
    const unsigned n, const int batch,
    const std::vector<std::vector<std::vector<unsigned>>> &cbuckets,
    const std::vector<Entry> &db) noexcept {
#ifdef G6K_MPI

  // If we're tracking unique sends, then do that first.
#ifdef MPI_TRACK_UNIQUES
  count_uniques(cbuckets, db);
#endif

  // This does all of the heavy lifting.
  dispatch_to_forward_and_gather_buckets<true>(
      n, batch, cbuckets, db, bgj1_buckets.get_sizes(batch),
      bgj1_buckets.get_comm(batch).get_comm());
#else
  return;
#endif
}

void MPIObj::deal_with_finished(const unsigned batch, const unsigned n,
                                const unsigned index, const unsigned size,
                                const unsigned buffer_index,
                                std::vector<Entry> &bucket) noexcept {
#ifdef G6K_MPI
  // We now need to unpack and copy over all of the serialised entries.
  // This takes the form of resizing the buckets and packing everything into the
  // right place.

  // First of all, we need to acquire the writes.
  // This just makes sure everything is appropriately synced.
  std::atomic_thread_fence(std::memory_order_acquire);

  // Next we need to work out how large the new bucket
  // will be.

  const auto &buffer = scratch_space[buffer_index].incoming_scratch;
  const auto nr_ranks = static_cast<unsigned>(number_of_ranks());
  const auto &bucket_pairs = get_bucket_pairs(batch);

  // The position of the elements in bucket_pairs is index * nr_ranks (you can
  // think of this as `sizes` but transposed).
  const auto position = index * nr_ranks;

  // We now need to know how many entries we need in bucket.
  // We add our own size in too, as we'll need to tack those on later.
  const auto bucket_size =
      size + std::accumulate(bucket_pairs.cbegin() + position,
                             bucket_pairs.cbegin() + position + nr_ranks,
                             uint64_t(0),
                             [](const uint64_t lhs, const bucket_pair &rhs) {
                               return lhs + rhs.size;
                             });
  // And now we resize based on that.
  // N.B This saves on memmoves: we don't care about what was in the bucket
  // before, which makes resizing cheaper.
  bucket.clear();
  bucket.resize(bucket_size);
  bucket.shrink_to_fit();

  // And now we just unpack based on the indices in bucket_pairs.
  unsigned used{};
  for (unsigned i = 0; i < nr_ranks; i++) {
    if (i == static_cast<unsigned>(rank))
      continue;
    const auto &pair = bucket_pairs[position + i];

    // N.B We need to scale the position by `n`, because the position is based
    // on the underlying type: however, the vector itself is full of ZTs.
    uint64_t pos = n * pair.pos;
    for (unsigned j = 0; j < pair.size; j++) {
      std::copy(buffer.cbegin() + pos, buffer.cbegin() + pos + n,
                bucket[used].x.begin());
      pos += n;
      ++used;
    }
  }

  // Now we decrement the reference count for this.
  dec_batch_use(buffer_index);
#endif
}

long double MPIObj::get_cpu_time(const std::clock_t time) const noexcept {
#ifdef G6K_MPI
  return MPIWrapper::get_cpu_time(comm, time);
#else
  return 0.0;
#endif
}

void MPIObj::set_extra_memory_used(const uint64_t used) noexcept {
#ifdef G6K_MPI
  struct rusage use {};
  getrusage(RUSAGE_SELF, &use);
  // N.B The 1000 here is to convert the KB reading from getrusage to
  // bytes.
  this->extra_memory = (1000 * use.ru_maxrss);
#else
  (void)used;
#endif
}

uint64_t MPIObj::get_extra_memory_used() const noexcept {
#ifdef G6K_MPI
  return MPIWrapper::get_extra_memory_used(comm, extra_memory);
#else
  return 0;
#endif
}

TimerPoint MPIObj::time_bgj1() noexcept {
#if defined(MPI_TIME) && defined(G6K_MPI)
  return timer.time(ContextChange::BGJ1);
#else
  __builtin_unreachable();
#endif
}

TimerPoint MPIObj::time_bdgl() noexcept {
#if defined(MPI_TIME) && defined(G6K_MPI)
  return timer.time(ContextChange::BDGL);
#else
  __builtin_unreachable();
#endif
}

uint64_t MPIObj::get_total_messages() noexcept {
#ifdef G6K_MPI
  return MPIWrapper::get_total_messages(comm);
#else
  return 0;
#endif
}

uint64_t MPIObj::get_total_bandwidth() noexcept {
#ifdef G6K_MPI
  return MPIWrapper::get_total_bandwidth(comm);
#else
  return 0;
#endif
}

uint64_t MPIObj::get_messages_for(const ContextChange type) noexcept {
#ifdef G6K_MPI
  return MPIWrapper::get_messages_for(type, comm);
#else
  return 0;
#endif
}

uint64_t MPIObj::get_bandwidth_for(const ContextChange type) noexcept {
#ifdef G6K_MPI
  return MPIWrapper::get_bandwidth_for(type, comm);
#else
  return 0;
#endif
}

uint64_t MPIObj::get_bgj1_center_bandwidth() noexcept {
#ifdef G6K_MPI
  return get_bandwidth_for(ContextChange::BGJ1_CENTER_SEND);
#else
  return 0;
#endif
}

uint64_t MPIObj::get_bgj1_buckets_bandwidth() noexcept {
#ifdef G6K_MPI
  return get_bandwidth_for(ContextChange::BGJ1_SEND_SIZE) +
         get_bandwidth_for(ContextChange::BGJ1_BUCKET_SEND);
#else
  return 0;
#endif
}

uint64_t MPIObj::get_bgj1_bandwidth_used() noexcept {
#ifdef G6K_MPI
  return get_bgj1_buckets_bandwidth() + get_bgj1_center_bandwidth() +
         get_bandwidth_for(ContextChange::INSERT) +
         get_bandwidth_for(ContextChange::ADHOC_INSERT);
#else
  return 0;
#endif
}

uint64_t MPIObj::get_bgj1_messages_used() noexcept {
#ifdef G6K_MPI
  return get_messages_for(ContextChange::BGJ1_BUCKET_SEND) +
         get_messages_for(ContextChange::BGJ1_CENTER_SEND) +
         get_messages_for(ContextChange::BGJ1_SEND_SIZE) +
         get_messages_for(ContextChange::INSERT) +
         get_messages_for(ContextChange::ADHOC_INSERT);
#else
  return 0;
#endif
}

void MPIObj::reset_bandwidth() noexcept {
#if defined(G6K_MPI) && defined(MPI_BANDWIDTH_TRACK)
  MPIWrapper::reset_bandwidth(comm);
#endif
}

#if defined(G6K_MPI) && defined(MPI_TRACK_UNIQUES)
void MPIObj::count_uniques(
    const std::vector<std::vector<std::vector<unsigned>>> &cbuckets,
    const std::vector<Entry> &db) noexcept {

  // Just iterate over the cbuckets and count how many are unique in each set.
  for (unsigned i = 0; i < number_of_ranks(); i++) {
    if (i == rank)
      continue;
    std::unordered_set<UidType> uniques;
    uint64_t size{};
    for (const auto &bucket_set : cbuckets[i]) {
      for (const auto &index : bucket_set) {
        uniques.insert(db[index].uid);
      }
      size += bucket_set.size();
    }

    unique_sends += uniques.size();
    nr_sends += size;
  }
}
#endif

#if defined(G6K_MPI)
std::array<uint64_t, 2> MPIObj::get_unique_ratio() noexcept {
#ifdef MPI_TRACK_UNIQUES
  return MPIWrapper::get_unique_ratio(unique_sends, nr_sends, comm);
#else
  return std::array<uint64_t, 2>{};
#endif
}
#endif

uint64_t MPIObj::get_adjust_timings() noexcept {
#if defined(G6K_MPI) && defined(MPI_TIME)
  const auto val = timer.get_time(ContextChange::EL) +
                   timer.get_time(ContextChange::ER) +
                   timer.get_time(ContextChange::IL) +
                   timer.get_time(ContextChange::SL_REDIST) +
                   timer.get_time(ContextChange::GROW_SMALL) +
                   timer.get_time(ContextChange::GROW_LARGE) +
                   timer.get_time(ContextChange::SHRINK) +
                   timer.get_time(ContextChange::SORT) +
                   timer.get_time(ContextChange::CHANGE_STATUS) +
                   timer.get_time(ContextChange::GSO_PP) +
                   timer.get_time(ContextChange::LOAD_GSO) +
                   timer.get_time(ContextChange::HISTO) +
                   timer.get_time(ContextChange::ORDERED_DB_SPLIT) +
                   timer.get_time(ContextChange::DB_SPLIT_UIDS) +
                   timer.get_time(ContextChange::DB_SIZE) +
                   timer.get_time(ContextChange::REDIST_DB) +
                   timer.get_time(ContextChange::RESERVE) +
                   timer.get_time(ContextChange::RECONCILE_DB) +
                   timer.get_time(ContextChange::BEST_LIFTS) +
                   timer.get_time(ContextChange::GET_DB) +
                   timer.get_time(ContextChange::SPLIT_DB) +
                   timer.get_time(ContextChange::LIFT_BOUNDS) +
                   timer.get_time(ContextChange::SL_NO_REDIST);
  timer.reset_time();
  return val;
#else
  return 0;
#endif
}

void MPIObj::write_stats(const unsigned n, const size_t dbs) noexcept {
#ifndef G6K_MPI
  return;
#elif defined(MPI_TRACK_UNIQUES) || defined(MPI_TRACK_BANDWIDTH) ||            \
    defined(MPI_TIME)
  assert(is_root());
  // We will load all of the various unique stats from the rest
  // of the world and then go from there. We write these in the same
  // order as in the constructor.
  const auto counts = get_unique_ratio();
  const auto bgj1_c_bandwidth = get_bgj1_center_bandwidth();
  const auto bgj1_b_bandwidth = get_bgj1_buckets_bandwidth();
  const auto bgj1_full_bandwidth = get_bgj1_bandwidth_used();
  const auto bgj1_messages = get_bgj1_messages_used();
  const auto cpu_time = get_cpu_time(std::clock_t(0));
  const auto extra_memory_used = get_extra_memory_used();
  const auto regular_memory =
      db_capacity(dbs) * (sizeof(Entry) + sizeof(CompressedEntry));

#ifdef MPI_TIME
  const auto bgj1_time = timer.get_time(ContextChange::BGJ1);
  const auto bdgl_time = timer.get_time(ContextChange::BDGL);
#else
  const auto bgj1_time = 0;
  const auto bdgl_time = 0;
#endif

  // N.B This call clears all of the timers used, so it also resets the bgj1
  // timer.
  const auto adjust_timings = get_adjust_timings();

  std::ofstream writer(filepath_for_track, std::ios_base::app);
  writer << n << "," << counts[0] << "," << counts[1] << "," << bgj1_c_bandwidth
         << "," << bgj1_b_bandwidth << "," << bgj1_full_bandwidth << ","
         << bgj1_messages << "," << cpu_time << "," << extra_memory_used << ","
         << regular_memory << "," << bgj1_time << "," << bdgl_time << ","
         << adjust_timings << std::endl;

  writer.close();
  reset_bandwidth();
#endif
}

void MPIObj::send_insertion_sizes(const unsigned batch,
                                  const unsigned n) noexcept {
#ifdef G6K_MPI
  auto &sizes = bgj1_buckets.get_sizes(batch);
  auto &block = tmp_insertion_buf[batch];

  std::fill(sizes.begin(), sizes.begin() + number_of_ranks(), 0);

  unsigned curr{};
  for (unsigned i = 0; i < number_of_ranks(); i++) {
    if (i == rank) {
      continue;
    }

    std::lock_guard<std::mutex> lock(block.buffers[i].lock);
    // N.B sizes are given in terms of the number of entries that are sent,
    // rather than in terms of the number of ZTs.
    sizes[i] = block.buffers[i].buffer.size() / n;
  }

  auto &communicator = bgj1_buckets.get_comm(batch);
  using T = std::remove_reference_t<decltype(sizes)>::value_type;

  TRACK(ContextChange::BGJ1_SEND_SIZE, sizeof(T) * (number_of_ranks() - 1));
  MPI_Ialltoall(MPI_IN_PLACE, 1, MPI_INT, sizes.data(), 1, MPI_INT,
                communicator.get_comm(), requests.size_requests(batch));
  bgj1_buckets.start_incoming_insertion_size(batch);
#else
  return;
#endif
}

void MPIObj::start_insertions(const unsigned batch, const unsigned n) noexcept {
#ifdef G6K_MPI
  auto &sizes = bgj1_buckets.get_sizes(batch);
  auto &buf = insertion_vector_bufs[batch];
  auto &block = tmp_insertion_buf[batch];
  size_t inc_size{}, out_size{};

  finished_sizes[batch] = false;

  for (unsigned i = 0; i < number_of_ranks(); i++) {
    if (i == rank) {
      scounts[i] = 0;
      sdispls[i] = 0;
      rcounts[i] = 0;
      rdispls[i] = 0;
      continue;
    }

    // Accumulate both the incoming and outgoing size, positions.
    {
      std::lock_guard<std::mutex> lock(block.buffers[i].lock);
      scounts[i] = block.buffers[i].buffer.size() / n;
      sdispls[i] = out_size;
      out_size += scounts[i];
    }

    rdispls[i] = inc_size;
    inc_size += sizes[i];
  }

  // Resize the buffers.
  buf.incoming_buffer.clear();
  buf.outgoing_buffer.clear();

  buf.incoming_buffer.resize(inc_size * n);
  buf.outgoing_buffer.resize(out_size * n);

  unsigned pos{};

  // Now copy over for the outgoing.
  for (unsigned i = 0; i < number_of_ranks(); i++) {
    if (i == rank)
      continue;
    std::lock_guard<std::mutex> lock(block.buffers[i].lock);
    std::copy(block.buffers[i].buffer.cbegin(), block.buffers[i].buffer.cend(),
              buf.outgoing_buffer.begin() + pos);
    pos += block.buffers[i].buffer.size();
    // Shrink the outgoing buffers back down.
    block.buffers[i].buffer.clear();
    block.buffers[i].buffer.shrink_to_fit();
  }

  TRACK(ContextChange::INSERT, buf.outgoing_buffer.size() * sizeof(ZT));
  auto &communicator = bgj1_buckets.get_comm(batch);

  MPI_Ialltoallv(buf.outgoing_buffer.data(), scounts.data(), sdispls.data(),
                 entry_type, buf.incoming_buffer.data(), sizes.data(),
                 rdispls.data(), entry_type, communicator.get_comm(),
                 requests.insertion_requests(batch));
  bgj1_buckets.start_incoming_insertion(batch);
#endif
}

void MPIObj::start_bdgl(const size_t nr_buckets_aim, const size_t blocks,
                        const size_t multi_hash) noexcept {
#ifdef G6K_MPI
  assert(is_root());
  MPIWrapper::start_bdgl(nr_buckets_aim, blocks, multi_hash, comm);
#endif
}

std::array<uint64_t, 3> MPIObj::receive_bdgl_params() noexcept {
#ifdef G6K_MPI
  assert(!is_root());
  return MPIWrapper::receive_bdgl_params(comm);
#else
  __builtin_unreachable();
#endif
}

std::vector<uint32_t> &
MPIObj::setup_bdgl_bucketing(const unsigned n, const size_t nr_buckets_aim,
                             const size_t bsize,
                             const size_t cdb_size) noexcept {
#ifdef G6K_MPI
  // This just sets up all of the auxiliary bits and pieces.
  bdgl_owned_buckets.resize(nr_buckets_aim);
  bdgl_nr_buckets = nr_buckets_aim;
  bdgl_bucket_size = bsize;
  bdgl_nr_completed = 0;
  nr_remaining_buckets.store(0);
  bdgl_batch_info.resize(bucket_batches);
  entry_type = Layouts::get_entry_type(n);

  completed.resize(number_of_ranks());
  std::fill(completed.begin(), completed.end(), false);

  const auto res =
      MPIWrapper::setup_owner_array(bdgl_owned_buckets, memory_per_rank);

  bdgl_bucket_map.resize(number_of_ranks());
  bdgl_used_buckets.resize(number_of_ranks());
  std::fill(bdgl_used_buckets.begin(), bdgl_used_buckets.end(), 0);

  for (unsigned i = 0; i < bdgl_bucket_map.size(); i++) {
    bdgl_bucket_map[i].clear();
    bdgl_bucket_map[i].reserve(res[i]);
  }

  for (unsigned i = 0; i < bdgl_owned_buckets.size(); i++) {
    bdgl_bucket_map[bdgl_owned_buckets[i]].emplace_back(i);
  }

  bdgl_used_buckets.resize(number_of_ranks());
  std::fill(bdgl_used_buckets.begin(), bdgl_used_buckets.end(), 0);

  if (bdgl_insert_pos.size() == 0) {
    bdgl_insert_pos =
        std::vector<padded_atom_int64_t>(bdgl_thread_entries.size());
  }

  for (unsigned i = 0; i < bdgl_insert_pos.size(); i++) {
    bdgl_insert_pos[i].store(cdb_size - i - 1);
  }

  bdgl_nr_outstanding = 0;

  bdgl_aux.cbuckets = std::vector<std::vector<std::vector<unsigned>>>(
      number_of_ranks(), std::vector<std::vector<unsigned>>(buckets_per_rank));
  bdgl_aux.sizes.resize(number_of_ranks() * buckets_per_rank);

  for (auto &v : bdgl_bucket_active) {
    v.store(0);
  }

  return bdgl_bucket_map[rank];
#else
  __builtin_unreachable();
#endif
}

void MPIObj::mark_as_replaced(const UidType removed) noexcept {
#ifdef G6K_MPI
  if (!owns_uid(removed)) {
    const auto owner = slot_map[UidHashTable::get_slot(removed)];
    std::lock_guard<std::mutex> lock(to_remove[owner].lock);
    to_remove[owner].buffer.emplace_back(removed);
  }

#endif
}

void MPIObj::mark_as_unused(const UidType unused) noexcept {
#ifdef G6K_MPI
  if (!owns_uid(unused)) {
    const auto owner = slot_map[UidHashTable::get_slot(unused)];
    // This is covered by the outer check, but just to be explicit.
    assert(owner != rank);
    // This lock is likely really contended in few node settings.
    std::lock_guard<std::mutex> lock(to_remove[owner].lock);
    to_remove[owner].buffer.emplace_back(unused);
  }
#endif
}

#ifdef G6K_MPI
template <bool is_root>
static auto product_lsh_impl(MPI_Comm comm,
                             const ProductLSH *const lsh = nullptr) noexcept {
  // Warning: this code is a little bit complicated.
  // Unlike the original AVX2 bucketer, we cannot assume that sending the seed
  // is necessarily enough here. This is because the Simd code may use a
  // different random number generator depending on if AES instructions are
  // available or not. We thus check this once and then, if appropriate,
  // forward the seed and other associated info directly.

  static int called = false;
  static int supports_aes = false;
  static int all_aes = false;
  if (!called) {
    __builtin_cpu_init();
    called = true;
    supports_aes = __builtin_cpu_supports("aes");
    // Collect all of the results.
    MPI_Allreduce(&supports_aes, &all_aes, 1, MPI_INT, MPI_LAND, comm);
  }

  if (!all_aes) {
    std::cerr
        << "Error: bdgl_broadcast_lsh not yet supported for non AES-NI machines"
        << std::endl;
    MPI_Abort(comm, 30);
  }

  Layouts::ProductLSHLayout tmp;
  auto type = Layouts::get_product_lsh_data_type_aes();
  if constexpr (is_root) {
    // Just broadcast the setup information.
    assert(lsh);
    tmp.n = lsh->n;
    tmp.blocks = lsh->blocks;
    tmp.code_size = lsh->codesize;
    tmp.multi_hash = lsh->multi_hash;
    tmp.seed = lsh->seed;

    MPI_Bcast(&tmp, 1, type, MPIWrapper::global_root_rank, comm);
    MPI_Type_free(&type);
    return;
  } else {
    MPI_Bcast(&tmp, 1, type, MPIWrapper::global_root_rank, comm);
    MPI_Type_free(&type);
    return ProductLSH(tmp.n, tmp.blocks, tmp.code_size, tmp.multi_hash,
                      tmp.seed);
  }
}
#endif

ProductLSH MPIObj::bdgl_build_lsh() noexcept {
#ifdef G6K_MPI
  return product_lsh_impl<false>(comm);
#else
  __builtin_unreachable();
#endif
}

void MPIObj::bdgl_broadcast_lsh(const ProductLSH &lsh) noexcept {
#ifdef G6K_MPI
  product_lsh_impl<true>(comm, &lsh);
#else
  return;
#endif
}

#ifdef G6K_MPI
template <int tag, typename T, typename F>
static void handle_hash_table_query(UidHashTable &table,
                                    std::vector<MPI_Request> &reqs,
                                    const unsigned comm_size, MPI_Comm comm,
                                    F &&process_func) noexcept {

  std::vector<T> inc_inserted;
  auto remaining_read = static_cast<int>(comm_size - 1);
  int finished;
  MPI_Status stat;

  constexpr auto type = Layouts::get_data_type<T>();
  assert(comm_size == reqs.size());

  while (remaining_read > 0) {
    MPI_Testall(comm_size, reqs.data(), &finished, MPI_STATUSES_IGNORE);
    MPI_Probe(MPI_ANY_SOURCE, tag, comm, &stat);
    int message_size;
    MPI_Get_count(&stat, type, &message_size);
    inc_inserted.resize(message_size);
    MPI_Recv(inc_inserted.data(), inc_inserted.size(), type, stat.MPI_SOURCE,
             tag, comm, MPI_STATUS_IGNORE);

    process_func(stat.MPI_SOURCE, inc_inserted);
    --remaining_read;
  }

  // Wait for the remaining ones to terminate (if any).
  MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}
#endif

void MPIObj::count_uniques(const std::vector<Entry> &bucket) noexcept {
#if defined(G6K_MPI) && defined(MPI_TRACK_UNIQUES)
  std::unordered_set<UidType> uniques;
  for (const auto &v : bucket) {
    uniques.insert(v.uid);
  }

  bucket_size += bucket.size();
  unique_entries += uniques.size();
#endif
}

void MPIObj::print_uniques(const unsigned iter) const noexcept {
#if defined(G6K_MPI) && defined(MPI_TRACK_UNIQUES)
  std::cerr << iter << double(bucket_size - unique_entries) / unique_entries
            << std::endl;
#endif
}

bool MPIObj::bdgl_gbl_uniques(const std::vector<Entry> &ents) {
#if defined(G6K_MPI)
  std::vector<UidType> elems(ents.size());
  std::unordered_set<UidType> set;
  for (unsigned i = 0; i < ents.size(); i++) {
    elems[i] = ents[i].uid;
    set.insert(elems[i]);
  }

  assert(set.size() == elems.size());

  bool were_same;

  if (rank == MPIWrapper::global_root_rank) {
    auto size = elems.size();
    std::vector<UidType> inc;

    for (unsigned i = 0; i < number_of_ranks(); i++) {
      if (i == rank) {
        continue;
      }
      MPI_Status stat;
      MPI_Probe(i, static_cast<int>(ContextChange::STATS_LONG), comm, &stat);
      int count;
      MPI_Get_count(&stat, Layouts::get_data_type<UidType>(), &count);
      inc.resize(count);
      MPI_Recv(inc.data(), inc.size(), Layouts::get_data_type<UidType>(), i,
               static_cast<int>(ContextChange::STATS_LONG), comm,
               MPI_STATUS_IGNORE);
      size += inc.size();
      for (auto &v : inc) {
        set.insert(v);
      }
    }

    were_same = (set.size() == size);
  } else {
    MPI_Send(elems.data(), ents.size(), Layouts::get_data_type<UidType>(),
             MPIWrapper::global_root_rank,
             static_cast<int>(ContextChange::STATS_LONG), comm);
  }
  MPI_Bcast(&were_same, 1, Layouts::get_data_type<bool>(),
            MPIWrapper::global_root_rank, comm);
  return were_same;
#else
  return true;
#endif
}

bool MPIObj::bdgl_inserts_consistent(const std::vector<Entry> &db,
                                     UidHashTable &hash_table) noexcept {
#ifdef G6K_MPI

  std::array<uint64_t, 2> elems{db.size(), hash_table.hash_table_size_safe()};
  MPI_Allreduce(MPI_IN_PLACE, elems.data(), elems.size(),
                Layouts::get_data_type<uint64_t>(), MPI_SUM, comm);
  if (is_root()) {
    std::cerr << "[Rank " << rank << "] nr_elems:" << elems[0]
              << " nr_entries:" << elems[1] << std::endl;
  }
  return elems[0] == elems[1];
#else
  return true;
#endif
}

void MPIObj::bdgl_gather_sizes(const std::vector<uint64_t> &sizes_) noexcept {
#ifdef G6K_MPI
  this->sizes.resize(number_of_ranks() * sizes_.size());
  std::copy(sizes_.cbegin(), sizes_.cend(),
            this->sizes.begin() + sizes_.size() * rank);
  MPI_Allgather(MPI_IN_PLACE, sizes_.size(), MPI_UINT64_T, this->sizes.data(),
                sizes_.size(), MPI_UINT64_T, comm);
#endif
}

void MPIObj::forward_and_gather_buckets_bdgl(
    const unsigned n, const int batch, const std::vector<uint32_t> &buckets,
    const std::vector<CompressedEntry> &cdb,
    const std::vector<Entry> &db) noexcept {
#ifdef G6K_MPI
  // Get the index of the scratch buffer we'll use.
  const auto index = get_unused_scratch_buffer(batch);

  // Now load all of the other bits.
  auto &scratch_vector = scratch_space[index];

  // The very first thing we'll do is make sure that the circumstance of
  // "we have nothing to do for ourselves" is never violated. We also use this
  // opportunity to initialise the various bits that we'll care about for
  // ourselves in the following loop (for reasons that will become clear).
  if (completed[rank]) {
    bdgl_batch_info[batch].pos = 0;
    bdgl_batch_info[batch].size = 0;
  } else {
    const auto to_process =
        std::min(buckets_per_rank,
                 bdgl_bucket_map[rank].size() - bdgl_used_buckets[rank]);
    bdgl_batch_info[batch].pos = bdgl_used_buckets[rank];
    bdgl_batch_info[batch].size = to_process;

    bdgl_used_buckets[rank] += to_process;
    nr_remaining_buckets.fetch_add(to_process, std::memory_order_relaxed);

    if (bdgl_used_buckets[rank] == bdgl_bucket_map[rank].size()) {
      completed[rank] = true;
      ++bdgl_nr_completed;
    }
  }

  std::fill(bdgl_aux.sizes.begin(), bdgl_aux.sizes.end(), 0);

  for (unsigned i = 0; i < number_of_ranks(); i++) {
    if (i == rank) {
      continue;
    }

    for (auto &bucket : bdgl_aux.cbuckets[i]) {
      bucket.clear();
    }

    // In this loop we have two main goals:
    // 1) We need to account for the elements that we'll receive from rank `i`,
    // and 2) We need to account for the elements that we'll send to rank `i`.
    // We already have all of the information we need for both.

    const auto to_send = std::min(buckets_per_rank, bdgl_bucket_map[i].size() -
                                                        bdgl_used_buckets[i]);

    // If there's actually work to do, then we need to build the cbuckets that
    // we'll send. We do that by copying over the outgoing vectors that we have
    // in our database.
    const auto send_offset = bdgl_used_buckets[i];
    uint64_t sending{};
    for (unsigned j = 0; j < to_send; j++) {
      // We have the outgoing size for this at sizes[rank*bdgl_nr_buckets + j
      // + send_offset].
      const auto size = sizes[rank * bdgl_nr_buckets + j + send_offset];
      // And the bucket's index is also similarly known.
      const auto bucket_id = bdgl_bucket_map[i][send_offset + j];
      // We now can just copy everything over.
      for (unsigned k = 0; k < size; k++) {
        bdgl_aux.cbuckets[i][j].emplace_back(
            cdb[buckets[bucket_id * bdgl_bucket_size + k]].i);
      }
      sending += size;
    }

    // Update the invariants.
    bdgl_used_buckets[i] += to_send;

    // We have to mark that we've finished it, too.
    if (bdgl_used_buckets[i] == bdgl_bucket_map[i].size() && !completed[i]) {
      completed[i] = true;
      ++bdgl_nr_completed;
    }

    // Now we need to deal with those that we have coming in. This is mainly
    // meant to handle mismatches where we may receive more buckets than we
    // send.
    // The number we're going to receive is already in
    // bdgl_batch_info[batch].size, and we know the starting position too (it's
    // in bdgl_batch_info[batch].pos).
    if (bdgl_batch_info[batch].size > 0) {
      const auto start =
          sizes.cbegin() + i * bdgl_nr_buckets + bdgl_batch_info[batch].pos;
      std::copy(start, start + bdgl_batch_info[batch].size,
                bdgl_aux.sizes.begin() + i * buckets_per_rank);
    }
  }

  dispatch_to_forward_and_gather_buckets<false>(n, batch, bdgl_aux.cbuckets, db,
                                                bdgl_aux.sizes, comm);
#endif
}

void MPIObj::forward_and_gather_all(const unsigned n,
                                    const std::vector<uint32_t> &buckets,
                                    const std::vector<CompressedEntry> &cdb,
                                    const std::vector<Entry> &db) noexcept {
#ifdef G6K_MPI
  if (bdgl_has_finished_contributing()) {
    return;
  }

  const auto nr_to_use = free_scratch.load(std::memory_order_relaxed);
  if (nr_to_use == 0 ||
      bdgl_bucket_active[wait_head].load(std::memory_order_relaxed) != 0) {
    return;
  }

  // If there's any left to use, then we will dispatch those batches, too.
  unsigned used{};

  // N.B The condition of the loop below is to stop us from issuing extra
  // batches when work has already finished: it prevents a race condition.
  do {
    forward_and_gather_buckets_bdgl(n, wait_head, buckets, cdb, db);
    ++used;
    wait_head = (wait_head == bucket_batches - 1) ? 0 : wait_head + 1;
  } while (used < nr_to_use && !bdgl_has_finished_contributing() &&
           bdgl_bucket_active[wait_head].load(std::memory_order_relaxed) == 0);

  free_scratch.fetch_sub(used, std::memory_order_relaxed);
  bdgl_nr_outstanding += used;
#endif
}

#ifdef G6K_MPI
template <int tag, typename F>
static void handle_hash_table_query(UidHashTable &table,
                                    std::vector<MPI_Request> &reqs,
                                    const unsigned comm_size, MPI_Comm comm,
                                    F &&process_func) noexcept {
  std::vector<UidType> inc_inserted;
  auto remaining_read = static_cast<int>(comm_size - 1);
  int finished;
  MPI_Status stat;

  constexpr auto type = Layouts::get_data_type<UidType>();

  assert(comm_size == reqs.size());

  while (remaining_read > 0) {
    MPI_Testall(comm_size, reqs.data(), &finished, MPI_STATUSES_IGNORE);
    MPI_Probe(MPI_ANY_SOURCE, tag, comm, &stat);
    int message_size;
    MPI_Get_count(&stat, type, &message_size);
    inc_inserted.resize(message_size);
    MPI_Recv(inc_inserted.data(), inc_inserted.size(), type, stat.MPI_SOURCE,
             tag, comm, MPI_STATUS_IGNORE);

    process_func(stat.MPI_SOURCE, inc_inserted);
    --remaining_read;
  }

  // Wait for the remaining ones to terminate (if any).
  MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}
#endif

void MPIObj::bdgl_run_queries(UidHashTable &hash_table,
                              thread_pool::thread_pool &pool) noexcept {
#ifdef G6K_MPI
  // In this function we're primarily concerned with answering global queries of
  // the form "Is this insertion something that we can handle?".
  //
  // This takes the form of sending all Uids to the appropriate node and reading
  // the results back in. This is rather trivial all things considered: we just
  // need to be careful to handle the case of "Is this an insertion we can
  // actually tolerate?".
  std::vector<std::vector<UidType>> uids(number_of_ranks());
  std::vector<std::mutex> uid_mutex(number_of_ranks());

  // For efficiency, we'll parallelise the insertions into uids by using local
  // buffers. This also prevents us from falling afoul of memory
  // inconsistencies.
  bdgl_nr_insertions = [this, &uids, &uid_mutex, &pool]() {
    std::atomic<unsigned> local_bdgl_nr_inserts{};
    pool.run([this, &uids, &uid_mutex, &local_bdgl_nr_inserts]() {
      const auto &thread_entry = bdgl_thread_entries[thread_pool::id];
      const auto &insertion_set = thread_entry.t_queue;
      if (insertion_set.size() == 0) {
        return;
      }

      std::vector<std::vector<UidType>> local_uids(number_of_ranks());
      for (const auto &e : insertion_set) {
        const auto owner = slot_map[UidHashTable::get_slot(e.uid)];
        if (owner == rank) {
          continue;
        }
        local_uids[owner].emplace_back(e.uid);
      }

      unsigned nr_outgoing{};

      for (unsigned i = 0; i < number_of_ranks(); i++) {
        if (i == rank)
          continue;
        std::lock_guard<std::mutex> lock(uid_mutex[i]);
        uids[i].insert(uids[i].end(), local_uids[i].cbegin(),
                       local_uids[i].cend());
        nr_outgoing += local_uids[i].size();
      }

      local_bdgl_nr_inserts.fetch_add(nr_outgoing, std::memory_order_relaxed);
    });
    return local_bdgl_nr_inserts.load();
  }();

  const auto comm_size = number_of_ranks();
  std::vector<MPI_Request> reqs(comm_size);
  constexpr auto tag = static_cast<int>(ContextChange::BDGL_QUERY);
  reqs[rank] = MPI_REQUEST_NULL;
  // N.B This needs to be allocated outside of pack_func, or we'll end up
  // serialising an address that may no longer exist by the time the send needs
  // to complete.
  UidType tmp{};

  const auto pack_func = [this, comm_size, &tmp,
                          tag](const std::vector<std::vector<UidType>> &to_send,
                               std::vector<MPI_Request> &reqs) {
    assert(to_send.size() == comm_size);
    assert(to_send.size() == reqs.size());
    constexpr auto type = Layouts::get_data_type<UidType>();
    for (unsigned i = 0; i < comm_size; i++) {
      if (i == rank) {
        continue;
      }

      const auto size = to_send[i].size();
      const auto *const ptr = (size == 0) ? &tmp : to_send[i].data();
      MPI_Isend(ptr, size, type, i, tag, comm, &reqs[i]);
    }
  };

  pack_func(uids, reqs);

  std::vector<std::vector<UidType>> rejected_inserts(comm_size);

  const auto filter_duplicates = [&rejected_inserts, &hash_table,
                                  this](const unsigned other_rank,
                                        const std::vector<UidType> &incoming) {
    for (const auto v : incoming) {
      assert(slot_map[UidHashTable::get_slot(v)] == rank);
      if (!hash_table.insert_uid(v)) {
        rejected_inserts[other_rank].emplace_back(v);
      }
    }
  };

  handle_hash_table_query<tag>(hash_table, reqs, comm_size, comm,
                               filter_duplicates);
  pack_func(rejected_inserts, reqs);

  const auto remove_rejected = [&hash_table,
                                this](const unsigned other_rank,
                                      const std::vector<UidType> &incoming) {
    for (const auto uid : incoming) {
      hash_table.erase_uid(uid);
    }
  };

  handle_hash_table_query<tag>(hash_table, reqs, comm_size, comm,
                               remove_rejected);
#endif
}

size_t MPIObj::get_bdgl_queue_size() noexcept {
#ifdef G6K_MPI
  return bdgl_nr_insertions;
#else
  return 0;
#endif
}

void MPIObj::bdgl_extract_entry(const unsigned thread_id, Entry &e,
                                const unsigned index,
                                const unsigned n) noexcept {
#ifdef G6K_MPI
  const auto start = index;
  const auto &thread_entry = bdgl_thread_entries[thread_id];

  assert(start < thread_entry.insert_db.size());
  assert(start + n <= thread_entry.insert_db.size());

  std::copy(thread_entry.insert_db.cbegin() + start,
            thread_entry.insert_db.cbegin() + start + n, e.x.begin());
#endif
}

void MPIObj::bdgl_remove_speculatives(UidHashTable &hash_table) noexcept {
#ifdef G6K_MPI
  const auto comm_size = number_of_ranks();
  // We remove any uids that we've added into our hash table (that weren't ours
  // to add) and forward those that we overwrote to the appropriate node.
  // Note that we only need to send those that we've overwritten: we've already
  // dealt with the ones we initially considered for insertion during the
  // earlier stage, and thus each node already knows of all potential
  // insertions.
  std::vector<MPI_Request> reqs(comm_size);
  reqs[rank] = MPI_REQUEST_NULL;
  UidType tmp{};

  constexpr auto tag = static_cast<int>(ContextChange::BDGL_REMOVE_DUP);
  static_assert(std::is_same_v<UidType, uint64_t>);

  for (unsigned i = 0; i < comm_size; i++) {
    if (i == rank)
      continue;

    std::lock_guard<std::mutex> tr_lock(to_remove[i].lock);
    const auto size = to_remove[i].buffer.size();
    const auto *const ptr = (size == 0) ? &tmp : to_remove[i].buffer.data();
    MPI_Isend(ptr, size, Layouts::get_data_type<UidType>(), i, tag, comm,
              &reqs[i]);
  }

  const auto remove_incoming = [&hash_table,
                                this](const unsigned,
                                      const std::vector<UidType> &inserted) {
    for (const auto v : inserted) {
      hash_table.erase_uid(v);
    }
  };

  handle_hash_table_query<tag>(hash_table, reqs, comm_size, comm,
                               remove_incoming);

  for (auto &v : to_remove) {
    std::lock_guard<std::mutex> lock(v.lock);
    v.buffer.clear();
    v.buffer.shrink_to_fit();
  }
#endif
}
