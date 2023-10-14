#include "dist_siever.hpp"
#ifndef DOCTEST_CONFIG_DISABLE
#include "doctest/extensions/doctest_mpi.h"
#endif
#include "mpi_cast.hpp"
#include "mpi_wrapper.hpp"

DistSiever::DistSiever(SieverParams p) noexcept : siever{p} {}

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("stop", 2) {
  // This test just checks that sending a stop message actually does the right
  // thing and stops the dist siever.
  std::vector<double> arr{1, 2, 3, 4, 5, 6, 7, 8, 9};
  if (test_rank == MPIWrapper::global_root_rank) {
    SieverParams p{};
    p.is_cython = true;
    p.comm = reinterpret_cast<uint64_t>(test_comm);
    Siever s{3, arr.data(), p};
    s.mpi.send_stop();
  } else {
    DistSiever dp{build_params(test_comm)};
    dp.run();
  }
}
#endif

#ifndef DOCTEST_CONFIG_DISABLE
MPI_TEST_CASE("EL", 2) {
  // This test just checks that sending a stop message actually does the right
  // thing and stops the dist siever.
  std::vector<double> arr{1, 2, 3, 4, 5, 6, 7, 8, 9};
  if (test_rank == MPIWrapper::global_root_rank) {
    SieverParams p{};
    p.is_cython = true;
    p.comm = reinterpret_cast<uint64_t>(test_comm);
    Siever s{3, arr.data(), p};
    s.mpi.broadcast_el(0);
    s.mpi.send_stop();
  } else {
    DistSiever dp{build_params(test_comm)};
    dp.run();
  }
}
#endif

void DistSiever::run() noexcept {
  auto &mp = siever.mpi;
  auto time = std::clock();

  while (true) {
    const auto status = mp.receive_command();
    // If we've received a stop command we just immediately bail.
    // This is here for clarity more than anything else.
    if (UNLIKELY(status[0] == ContextChange::STOP)) {
      return;
    }

    // Otherwise, we look at the first entry. This tells us what sort of message
    // we've received from the root.
    // Broadly speaking, these messages are grouped as follows:
    // (1) EL/ER/SL/CHANGE_STATUS/SORT/HISTO. These messages receive everything
    // they need
    //     in the initial receive_command call, and so we can just delegate to
    //     those directly.
    // (2) LOAD_GSO/GSO_PP. These require either reading a new vector (LOAD_GSO)
    // or an entirely
    //     new matrix (GSO_PP).
    // (3) BGJ1. This message says we need to start sieving. We need to read
    //     the alpha value.
    // (4) IL. This message says we need to do an initialize_local call. The
    // initial message contains the ll_ parameter,
    //         but we need to read some additional data to recover l_ and r_.
    switch (static_cast<ContextChange>(status[0])) {
    case ContextChange::EL:
      siever.extend_left(status[1]);
      break;
    case ContextChange::ER:
      siever.extend_right(status[1]);
      break;
    case ContextChange::SL_REDIST:
      siever.shrink_left(status[1], true);
      break;
    case ContextChange::SL_NO_REDIST:
      siever.shrink_left(status[1], false);
      break;
    case ContextChange::IL: {
      // The ll_ argument is stored in status[1], so we read the rest.
      const auto l_and_r = mp.receive_l_and_r();
      // The layout here is guaranteed by the API.
      siever.initialize_local(status[1], l_and_r[0], l_and_r[1]);
      break;
    }
    case ContextChange::SORT:
      siever.parallel_sort_cdb();
      break;
    case ContextChange::CHANGE_STATUS:
      siever.switch_mode_to(static_cast<Siever::SieveStatus>(status[1]));
      break;
    case ContextChange::GSO_PP: {
      std::vector<long> M;
      const auto aux = mp.receive_gso_update_postprocessing(M, siever.n);
      siever.gso_update_postprocessing(aux[0], aux[1], M.data(), aux[2]);
      break;
    }
    case ContextChange::LOAD_GSO: {
      std::vector<double> mu;
      // Here the full size is stored in the second entry of status.
      mp.receive_gso_no_header(status[1], mu);
      siever.load_gso(static_cast<unsigned>(sqrt(status[1])), mu.data());
      siever.r = sqrt(status[1]);
      siever.n = siever.r - siever.l;
      break;
    }
    case ContextChange::BGJ1: {
      const auto alpha = mp.receive_alpha();
      siever.bgj1_sieve(alpha);
      break;
    }
    case ContextChange::BDGL: {
      const auto params = mp.receive_bdgl_params();
      siever.bdgl_sieve(params[0], params[1], params[2]);
      break;
    }
    case ContextChange::HISTO:
      siever.recompute_histo();
      break;
    case ContextChange::DB_SIZE:
      if (status[1] == 0) {
        mp.db_size(siever.db_size());
      } else {
        mp.db_capacity(siever.db_size());
      }
      break;
    case ContextChange::GROW_SMALL:
      siever.grow_db(status[1], 0);
      break;
    case ContextChange::GROW_LARGE:
      siever.grow_db(status[1], 1);
      break;
    case ContextChange::RESERVE:
      siever.reserve(status[1]);
      break;
    case ContextChange::SHRINK:
      // Everything (all I/O etc) is handled inside
      // the shrink_db call.
      // NOTE: status[1] == 1 if the global DB size is not 0, and
      // 0 otherwise.
      mp.shrink_db(status[1], siever);
      break;
    case ContextChange::RESET_STATS:
      siever.reset_stats();
      break;
    case ContextChange::GET_CPU_TIME: {
      const auto now = std::clock();
      mp.get_cpu_time(now - time);
      time = now;
      break;
    }
    case ContextChange::GET_EXTRA_MEMORY:
      mp.get_extra_memory_used();
      break;
    case ContextChange::MESSAGES_FOR:
      mp.get_messages_for(static_cast<ContextChange>(status[1]));
      break;
    case ContextChange::BANDWIDTH_FOR:
      mp.get_bandwidth_for(static_cast<ContextChange>(status[1]));
      break;
    case ContextChange::TOTAL_MESSAGES:
      mp.get_total_messages();
      break;
    case ContextChange::TOTAL_BANDWIDTH:
      mp.get_total_bandwidth();
      break;
    case ContextChange::GET_UNIQUE_RATIO:
      mp.get_unique_ratio();
      break;
    case ContextChange::RESET_BANDWIDTH:
      mp.reset_bandwidth();
      break;
    default:
      // Do not convert this to a ContextChange: it'll cause a crash if it can't
      // be printed.
      std::cerr << "Received:" << static_cast<int>(status[0])
                << "but didn't know what to do with it. " << std::endl;
      assert(false);
    }
  }
}
